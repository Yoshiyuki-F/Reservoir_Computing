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

float32 は禁止（精度がガタ落ちするので）→ `JaxF64` で物理的に弾く

## 5. ヘキサゴナルアーキテクチャ — Import境界の物理的制約

### Domain分類ルール
| Domain | Import許可 | 例                                                   |
|--------|-----------|-----------------------------------------------------|
| **NUMPY** | `numpy` のみ | `preprocessing.py`, `data_prep.py`, `loaders.py`    |
| **JAX** | `jax` のみ | `projection.py`, `models/nn/*.py`, `aggregation.py` |
| **MAPPER** | 両方OK | `batched_compute.py`, `types.py' `strategies.py`     |
| **PURE** | どちらもなし | `config.py`, `identifiers.py`, `presets.py`         |

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

### JAX x64 Initialization Logic (The Gatekeeper Pattern)
Investigation revealed that JAX 0.9 config locking is extremely sensitive to import order.
Simply setting environment variables in `__init__.py` or `main.py` proved insufficient if `jax` was imported elsewhere (e.g., via `lint_imports` scanning or other utilities).

**Solution:**
The file `src/reservoir/utils/gpu_utils.py` acts as the **Gatekeeper**.
- It is called by `main.py` *before* any heavy computation.
- It explicitly executes `jax.config.update("jax_enable_x64", True)` **immediately before** checking devices.
- This ensures x64 is enabled exactly when the backend is initialized, overriding any default "float32" state that might have leaked.

**Rule:**
- **Do not remove** the `jax.config.update` call in `gpu_utils.py`. It is not redundant; it is the effective enforcement point.
- `main.py` and `__init__.py` still set `os.environ` as a best practice, but `gpu_utils.py` guarantees it.



# ==========================================
# 禁止ルール
# ==========================================

# 1. 厳格な型エイリアスの定義（AnyやUnionは一切禁止）
JaxF64 = Float64[jax.Array, "..."] 
NpF64 = Float64[np.ndarray, "..."]
のようにtypes.py 定義することでFloat64という形タイプをJaxのそれだと矯正します。そうすることでJaxTypingはtypes.py以外ではImportされないはず。
Float64, Float32 Float などの型エイリアスも禁止。JaxF64　やNpF64を徹底する。
だからといって型ヒントを無視するのは禁止。例えば、def vpt_score(y_true, y_pred, threshold: float ) -> int: のように、型ヒントが実際の値と乖離している場合は即修正するルールを徹底する。

# 2. Ctrl+F で Any, Union, などの曖昧な型ヒントがないか全ファイルを検索して、もしあれば即修正するルールを徹底する。Anyは引数の**args, **kwargsでのみ許可する。
🚫 厳密に禁止される「逃げ道」型:Importさえ禁止
Any
object（型ヒントとしての使用は一切禁止）
Union （FailSafeとしての両対応は禁止）
Type 
Optional

曖昧な Callable
✅ 許可される型ヒント（ホワイトリスト）:
引数や戻り値の型ヒントには、以下のいずれかしか使用してはならない。
プリミティブ型: int, float, str, bool
types.py で定義された厳格なエイリアス: JaxF64, NpF64
プロジェクト内で定義された基底クラス（ABC / Protocol）: BaseConfig, ModelConfig など
標準コレクション（内部の型も厳格に指定すること）: List[str], Tuple[int, ...] など（※ List[object] は禁止）
# 3. np.asarray jnp.asarray やnp.array も禁止。Mapper層以外でjnp.asarrayを使っているファイルがあれば即to_jax_f64やto_np_f64に修正するルールを徹底する。(これらもType.pyに定義済み)
nd.array
jnd.array も禁止

# 4. UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
この警告は出るべきではない。だからといって警告を無視するようなコードを書くのは禁止。もしこの警告が出るコードがあれば、即修正するルールを徹底する。

# 6. lint_imports.py もしくは同様のスクリプトを作成して、
     両方np とjnpimportしているファイルがないか
     上に書いてあるTypeが使われているか
全ファイルをチェックTestを追加する。

# 7. from reservoir.core.types import NpF64, JaxF64, to_jax_f64  これはおかしい。両方importしているのはMapper層だけなので、これもMapper層以外のファイルで見つけたら即修正するルールを徹底する。
禁止　
[Union[NpF64, JaxF64]
to_np_f64(train_pred_cached) if not isinstance(train_pred_cached, np.ndarray) else train_pred_cached　（わざわざ型指定したのにIfで逃げるのはNG)
import NpF64, JaxF64 （Mapper層以外で両方importするのは禁止。Mapper層以外ではどちらか一方だけをimportするルールを徹底する）
# 8.  Callable の厳格化ルール
❌ 禁止される書き方（曖昧）
Callable (引数・戻り値の省略は即アウト)
Callable[..., Any] (... による引数の省略や、戻り値の Any への逃げ)
Callable[[jnp.ndarray], jnp.ndarray] (生の配列型の使用は前述のルール違反)

✅ 許可される書き方（厳格）
必ず types.py で定義したエイリアス（JaxF64, NpF64 等）を用いて、「入力の数・型」と「出力の型」を完全に明記することを強制します。
Callable[[JaxF64], JaxF64] (JAX配列を受け取り、JAX配列を返す)
Callable[[NpF64, int], NpF64] (Numpy配列と整数を受け取り、Numpy配列を返す)
🚫 禁止事項
astype() による遅延キャストの禁止: パイプラインの途中で astype() を使用して型を合わせることは、メモリの重複コピーを引き起こすため固く禁じる。

np.copy() の禁止: 同様の理由で、明示的なディープコピーも原則禁止。

✅ 強制される実装パターン（Fail Fast & Contract）
Data Loaderの責任:
データローダー（loaders.py）の時点で、必ず最初から np.float64 としてロードまたは生成しなければならない。後段のレイヤーにキャストの責任を押し付けてはならない。

アサーションによる防御 (Fail Fast):
受け取る側の関数（Preprocessingなど）では、astype() で変換してあげるのではなく、**型が違えば即座にクラッシュさせる（Fail Fast）**こと。

# 9.
🚫 Numpy Host Domain におけるメモリ操作の厳格な禁止事項 (Memory Allocation Anti-Patterns)
エージェントはコード生成時、以下の「暗黙のメモリ倍増・肥大化」を引き起こす書き方を絶対に避けてください。
astype() による遅延キャスト:
❌ 禁止: X_arr = X.astype(np.float64) (RAM上のコピーが倍増する)
✅ 強制: Data Loaderの段階で dtype=np.float64 として生成し、後段はアサート (assert X.dtype == np.float64) で防ぐ。
np.eye()[labels] によるOne-hotエンコーディング:
❌ 禁止: 中間的に巨大な単位行列をアロケーションするため、クラス数が多いとOOMで即死する。
✅ 強制: np.zeros() でターゲット配列を確保し、arr[np.arange(N), labels] = 1.0 のようにインプレースでインデックス参照して書き込む。
np.copy() またはスライスによる不要な複製:
❌ 禁止: 計算途中の X_copy = np.copy(X) や X_new = X[:]
✅ 強制: 前処理（スケーリング等）はすべてインプレース演算（X -= mean 等）で行い、参照を維持したままパイプラインを流す。

#10. 
「型パラメータのない素の dict や list の使用は禁止。必ず dict[K, V] のように中身を明示すること」しかしAnyは禁止

#11.
models/ 
ridge などはJax強制

🚫 古い型ヒント（typing module）の禁止:
Python 3.10+ のモダン構文を強制する。
typing.Optional[X] の使用は禁止。必ず X | None を使用すること。
typing.Union[X, Y] の使用は禁止。必ず X | Y を使用すること。
typing.Dict, typing.List, typing.Type 等の大文字コレクションは禁止。組み込みの dict, list, type を使用すること。
type :ignore 




**HOW TO CHECK**
uv run pyrefly check
uv run python scripts/lint_imports.py
uv run ruff check