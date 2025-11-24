# Repository Guidelines

## Project Structure & Module Organization
- Core library: `src/core_lib/` (pure model + data logic).
- Application entrypoints: `cli/` (e.g. `cli/main.py` for `reservoir-cli`).
- Pipelines: `pipelines/` (experiment workflows, plotting, orchestration).
- Utilities: `src/core_lib/utils/` (GPU helpers, metrics, preprocessing).
- Tests: `tests/` (`test_*.py` under `tests/core_lib` and `tests/pipelines`).
- Presets: `presets/` (JSON configs for datasets/models/training/experiments).
- Scripts: `scripts/` (CUDA setup, rebuild helpers). Docs in `docs/`, example outputs in `outputs/`.

## Build, Test, and Development Commands
- Install deps (Python 3.13+): `uv sync`
- Run CLI: `uv run reservoir-cli --help`
- CPU tests: `uv run poe test` (equivalent: `uv run pytest -q`)
- GPU smoke test (skips if no GPU): `uv run poe test-gpu`
- GPU demos (CUDA 12): `uv run poe demo-sine-gpu` or `uv run poe demo-lorenz-gpu`

## Coding Style & Naming Conventions
- Follow PEP 8 with 4‑space indentation and type hints.
- Naming: modules/functions/variables `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Docstrings: concise summary + args/returns; prefer NumPy style where helpful.
- Keep files focused; colocate helpers in `utils/`; core RC logic in `reservoir/`.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/` and name `test_*.py` with functions `test_*`.
- Add unit tests for new logic and edge cases; prefer deterministic seeds.
- GPU-specific checks belong in `tests/test_gpu_*.py`; guard heavy tests with markers if needed.
- Run locally: `uv run pytest -q`; for CUDA paths use Poe tasks above.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤72 chars). Examples: `add poe tasks`, `update structure`, `fix metrics edge case`.
- PRs: clear description, rationale, and scope; link issues. Include:
  - What changed and why
  - How to test (commands used)
  - Screenshots/plots for demos when relevant
  - Notes on GPU/CUDA requirements

## Security & Configuration Tips
- CUDA 12 required for GPU: ensure drivers match. Poe tasks set `JAX_PLATFORMS=cuda` and disable XLA preallocation.
- Avoid committing large artifacts; write outputs to `outputs/` and add to `.gitignore` if new patterns arise.

## TODO: Reservoir Computingの入力プロジェクション最適化

### 概要
現在の `_reservoir_step` 内で行われている `W_in` と入力データの行列積を、`jax.lax.scan` ループの外に出して事前計算（Pre-computation）する "W_in stage" を追加する。

### タスク
1. Pre-computation (W_in stage):
   - `jax.lax.scan` を呼び出す直前に、入力時系列データ全体へ `W_in` を適用して `projected_inputs` を作成する。
   - 計算式: `projected_inputs = jnp.dot(input_sequence, W_in.T)`
   - 期待形状: `(Time, N_in)` -> `(Time, N_res)`
2. Step関数の修正:
   - `_reservoir_step` の引数 `input_data` は生データではなく `projected_input` (100次元) が渡されるようにする。
   - 関数内での `jnp.dot(self.W_in, input_data)` 計算を削除し、引数をそのまま `input_contribution` として使用する。

### 目的
- ループごとの行列ベクトル積を排除し、JAXのコンパイル最適化と実行速度を向上させる。

### JAXでの実装変更案
クラスのメソッド構成を想定し、呼び出し元（`__call__` や forward に相当）と、修正された `_reservoir_step` を記述する。

#### 修正後の `_reservoir_step`
```python
def _reservoir_step(self, carry, projected_input):
    """
    reservoirの1ステップを実行します（JAX scan用）。

    Args:
        carry: (state, key) のタプル
        projected_input: W_in @ u(t) 済みのベクトル (shape: n_hidden_layer)
    """
    state, key = carry
    key, subkey = random.split(key)

    noise = random.normal(subkey, (self.n_hidden_layer,), dtype=jnp.float64) * self.noise_level
    res_contribution = jnp.dot(self.W_res, state)
    input_contribution = projected_input

    pre_activation = res_contribution + input_contribution + noise
    new_state = (1 - self.alpha) * state + self.alpha * jnp.tanh(pre_activation)

    return (new_state, key), new_state
```

#### 呼び出し元の変更イメージ（W_in stageの追加）
```python
def __call__(self, input_sequence, key):
    """
    Args:
        input_sequence: (Time, N_in) 例: (28, 28)
        key: PRNGKey
    """
    # --- W_in stage (事前計算) ---
    projected_inputs = jnp.dot(input_sequence, self.W_in.T)

    init_state = jnp.zeros((self.n_hidden_layer,), dtype=jnp.float64)
    init_carry = (init_state, key)

    (final_state, _), states = jax.lax.scan(
        self._reservoir_step,
        init_carry,
        projected_inputs
    )

    return states
```
