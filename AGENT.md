# Reservoir Pipeline Architecture Rules

## Data Domain & Array Lifecycle (Numpy vs JAX)

To prevent severe System RAM OOM (Out-Of-Memory) issues caused by JAX's asynchronous dispatch and immutable array copying, the pipeline strictly separates array forms into **Host Domain** (Numpy) and **Device Domain** (JAX).

**Do NOT use `jnp.asarray` or `import jax.numpy as jnp` anywhere in the frontend.** 

### 1. Host Domain (System RAM / Frontend)
All data that holds the entire dataset MUST remain `numpy.ndarray` (specifically `np.float64`).
- **Step 1: Data Loaders (`reservoir/data/loaders.py`)** 
  - Must return `np.ndarray`.
  - Do not initialize JAX arrays here.
- **Step 2: Preprocessing (`reservoir/layers/preprocessing.py`)**
  - Must assert input is `np.ndarray`.
  - Must perform modifications strictly **in-place** (`arr -= mean`, `arr /= std`) to avoid RAM duplication. 
- **Step 6.5: Target Alignment & Storage (`reservoir/pipelines/components/*.py`)**
  - DataCoordinator and FrontendContext must hold only Numpy arrays.

### 2. Device Domain (VRAM / JIT Compiled Execution)
All core neural mechanics MUST operate on JAX (`jnp.ndarray`).
- **Step 3 (Projection), Step 5 (Model/Reservoir), Step 6 (Feature Extraction)**
  - These layers define pure `jnp` functions. They never handle the entire dataset at once.

### 3. The Strict Boundary (`utils/batched_compute.py`)
The **only permitted place** to cast data between Numpy and JAX is inside `batched_compute()`.
This utility acts as the gateway:
1. **Numpy -> JAX (Enter Device):** `batch_jax = jnp.asarray(batch_data, dtype=jnp.float64)`
2. **Execute:** `batch_out_jax = step(batch_jax)`
3. **JAX -> Numpy (Exit Device):** `result_array[start:end] = np.asarray(batch_out_jax)`

By adhering to this boundary, JAX never allocates memory for the full dataset, preventing the 32GB RAM limit from being exceeded.


Alias FailSafe禁止　
Callable[[jnp.ndarray], jnp.ndarray], inputs: Union[np.ndarray, jnp.ndarray],　やAnyなども禁止。どっちかのはず

Any is allowed just for args**
だからといってType無視は禁止（だめな例：def vpt_score(y_true, y_pred, threshold: float = 0.4) -> int:）
NPかJNPどちらか。どちらもImportしているファイルはそれを変えるためのMapper。

### 4. 厳格な型監視 (Runtime Type Checking) -- 今後の実装予定
型ヒントをごまかす `Union[np.ndarray, jnp.ndarray]` や `Any` は禁止とし、**Mapper層（`batched_compute`等）で境界を明確に分ける**設計を強制するため、今後は `beartype` と `jaxtyping` をデファクトスタンダードとして導入します。

1. **Beartypeによるランタイム強制:**
   - 動的型チェッカーとして関数呼び出し時に型を監視し、`process_jax(x: jnp.ndarray)` に `np.ndarray` が渡された瞬間に即座に例外（`BeartypeCallHintParamViolation`）を発生させます。
   - 「型ヒントと実際の値が違うことによる暗黙のバグ」をゼロにします。
2. **Jaxtypingによる厳密な配列定義:**
   - 生の `jnp.ndarray` ではなく、`jaxtyping` を組み合わせて「NumPyかJAXか」「次元数」「データ型（float64等）」まで厳密に監視するルールとします。
   - 例: `Float[ndarray, "batch time features"]` vs `Float[Array, "batch time features"]`
3. **静的型チェック（Mypy/Pyright）との併用:**
   - 実行時エラー（Beartype）に加えて、エディタ上でも静的チェッカーによる警告を出せるよう併用します。

float32 は禁止（精度がガタ落ちするので）→ `Float64[np.ndarray, "*dims"]` で物理的に弾く

## 5. ヘキサゴナルアーキテクチャ — Import境界の物理的制約

### Domain分類ルール
| Domain | Import許可 | 例 |
|--------|-----------|-----|
| **NUMPY** | `numpy` のみ | `preprocessing.py`, `data_prep.py`, `loaders.py` |
| **JAX** | `jax` のみ | `projection.py`, `models/nn/*.py`, `aggregation.py` |
| **MAPPER** | 両方OK | `batched_compute.py`, `metrics.py`, `strategies.py` |
| **PURE** | どちらもなし | `config.py`, `identifiers.py`, `presets.py` |

> `jaxtyping`/`beartype` は型アノテーションツールなのでNUMPYドメインでも使用可。

### 物理的強制 (Lint Script)
```bash
uv run python scripts/lint_imports.py
```
Mapper登録外のファイルで両方importしたら即エラー(exit code 1)。
CI/pre-commitフックに組み込み可能。

## CRITICAL: JAX 0.9 x64 起動方法

### 正しい起動コマンド
```bash
uv run python -m reservoir.cli.main --model fnn --dataset mnist
```

### 絶対に使ってはいけないコマンド
```bash
# NG: reservoir-cli (pyproject.toml entry point)
# → JAX x64が無効化され、全演算がfloat32に劣化する
reservoir-cli --model fnn --dataset mnist
```

### 根本原因 (JAX 0.9 Breaking Change)
JAX 0.9 は `JAX_ENABLE_X64` をプロセス起動時に読み取り、以降ロックする。
`pyproject.toml` の `[project.scripts]` が生成するエントリポイントスクリプトは、
`reservoir/__init__.py` の `os.environ["JAX_ENABLE_X64"] = "True"` が実行される前に
JAXバックエンドを初期化してしまい、x64=False でロックされる。

`uv run python -m` は正しいモジュール解決順序を保証するため、
`reservoir/__init__.py` → env var設定 → JAX import の順序で実行される。

# ==========================================
# 1. 厳格な型エイリアスの定義（AnyやUnionは一切禁止）
# ==========================================
JaxF64 = Float64[jax.Array, "..."] 
もしくは
NpF64 = Float64[np.ndarray, "..."]
# ArrayはデフォルトでJAX/NumPy汎用だが、JAX領域では実質JAX専用として扱う
のように定義することでFloat64という形タイプをJaxのそれだと矯正します