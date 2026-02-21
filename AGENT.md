# Reservoir Pipeline Architecture Rules

### 3. The Strict Boundary (`utils/batched_compute.py`)
The **only permitted place** to cast data between Numpy and JAX is inside `batched_compute()`.
This utility acts as the gateway:
1. **Numpy -> JAX (Enter Device):** `batch_jax = jnp.asarray(batch_data, dtype=jnp.float64)`
2. **Execute:** `batch_out_jax = step(batch_jax)`
3. **JAX -> Numpy (Exit Device):** `result_array[start:end] = np.asarray(batch_out_jax)`

float32 は禁止（精度がガタ落ちするので）→ `JaxF64` で物理的に弾く

### 物理的強制 (Lint Script)
```bash
uv run ruff check
uv run python scripts/lint_imports.py
uv run pyrefly check
```

上から順に、エラーが０になるまで実行してください。コードスタイル、型ヒントの厳格化、そして両方のインポート禁止をチェックするためのスクリプトです

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
Python 3.13 のモダン構文を強制する。
typing.Optional[X] の使用は禁止。必ず X | None を使用すること。
typing.Union[X, Y] の使用は禁止。必ず X | Y を使用すること。
typing.Dict, typing.List, typing.Type 等の大文字コレクションは禁止。組み込みの dict, list, type を使用すること。
type :ignore 

# 12. 🚫 Rule: 内部ロジックでの isinstance 防衛の禁止 (Trust the Type System)

PyreflyとBeartypeによって型は完全に保証されているため、「念のため」の isinstance や type() == による型チェック（Defensive Programming）を内部ロジックで行うことを固く禁ずる。

例外が許されるのは「意図的な Union 型（例: int | str）を分岐処理（Type Narrowing）する時」のみである。

外部からのデータ（JSONやユーザー入力）は、システム境界（MapperやConfigパース時）で一度だけバリデーションし、内部（ReportingやPipeline）には一切の型チェックを持ち込まないこと。

# ❌ 開発環境でしか守ってくれない「偽物の盾」
assert result.ndim == 2, f"Aggregation output must be 2D, got {result.shape}"

# ✅ 本番環境でも絶対にすり抜けを許さない「真の盾」
if result.ndim != 2:
    raise ValueError(f"Aggregation output must be 2D, got {result.shape}")
