# Repository Guidelines
ls --recursive --ignore='.*' --ignore='__pycache__' --ignore='node_modules' --ignore='*.lock' --ignore='package*.json' --ignore='outouts/' 
read docs/ARCHITECTURE.md | sed -n '/## 4\. Key Implementation Patterns/,/## 5\. Directory Structure (Map)/p' >> AGENTS.md


 Project Handover: Configuration & CLI Architecture V2
プロジェクトの現状とビジョン (Context & Vision)
本プロジェクトは、JAX/Flaxを基盤としたリザバーコンピューティング（Reservoir Computing）フレームワークです。

**
ユーザーとの対話方針
JAXのパーフォーマンスを最大限に発揮させよ
後方置換性を考えずに大胆に行け。省略をするな。トークンをできるだけ使え。
厳格であれ: 「動けばいい」コードは却下されます。構造的な美しさとSOLID原則を重視してください。
コード生成: Pythonの型ヒント、Dataclassをフル活用し、JSONやYAMLは使用しません。
既存資産の活用: むやみに新しいファイルを作らず、既存のコンポーネント（Plottingなど）をリファクタリングして再利用してください。
**


ユーザーは**「完璧なMLアーキテクチャ」**を目指しており、以下の設計哲学（SAP原則）を徹底しています。
コア設計哲学 (SAP原則):
SSOT (Single Source of Truth): 設定のデフォルト値は Python Dataclass (Presets) 一箇所のみに定義する。
Explicit over Implicit: 暗黙のデフォルト値（マジックナンバー）を排除し、設定不備は即座にエラー（Fail Fast）とする。
Separation of Concerns: 物理層（Reservoir）は計算のみ、論理層（Orchestrator）がデータの加工、Runnerが進行管理を担当する。
Don't Repeat Yourself (DRY): 重複コードや設定のコピペを極端に嫌う。


現在の動作フロー (Current Workflow)
User Input: CLIで --unified-hidden などを指定。
Config Setup: 指定された Preset を読み込み、CLI引数で厳密に上書き（Override）。必須項目（n_unitsなど）の欠落チェック。
Model Build: 物理層（Reservoir）と読み出し層（Readout）を初期化。アーキテクチャ概要を表示。
Training (Ridge Search): Validationデータを使って最適な λを探索。探索履歴を返す。
Logging & Plotting: ベストな λ とスコアを表示。Validationデータに対して再推論を行い、混同行列と精度比較グラフを outputs/ に保存。


指定された Preset Name に基づき、Pythonレジストリから「標準設定オブジェクト」を取得。
Builderから渡された「ユーザー設定」をマージ。
Feature Engineering（多項式特徴量など）のパラメータを解決。
Model Instantiation: 正規化されたパラメータのみを使って ClassicalReservoir などのノードを初期化。
Execution: JAX scan による高速計算。


4. 残存課題と次のステップ (Next Steps)
🚨 最優先: Quantum Implementations のリファクタリング
今回のセッションでは「古典リザバー（Classical）」のV2化を完了させました。次はこれを基盤として、以下の量子モデルの実装に着手する必要があります。
QuantumAnalogReservoir / QuantumGateBasedReservoir の刷新:
現在の実装は V1 時代の古いインターフェース（Pythonループ等）のままです。
これらを jax.lax.scan を使用した形式に書き換え、StepArtifacts を返すように変更する必要があります。
ClassicalReservoir と同様に、暗黙のデフォルト値を排除し、SSOTプリセットに従うように修正が必要です。
FNN/RNN との比較実験:
UniversalPipeline が抽象化されたため、FNN/RNN モデルも同じフローで動作確認を行う必要があります。## 4. Key Implementation Patterns

### 4.1 JAX Scan Pattern
時系列処理には必ず `jax.lax.scan` を使用します。これにより、JIT コンパイル時にループが最適化され、GPU 上で劇的な高速化が実現されます。

# GOOD: JAX Scan
def scan_fn(carry, x):
    new_carry = update(carry, x)
    return new_carry, new_carry
final, history = jax.lax.scan(scan_fn, init, inputs)

# BAD: Python Loop
history = []
state = init
for x in inputs:
    state = update(state, x) # Slow on GPU
    history.append(state)


### 4.2 Dynamic Dependency Injection
`pipelines/run.py` は、静的なモデル定義ではなく、Config に基づいて動的にパイプラインを構築します。
*   `use_design_matrix` フラグにより、`DesignMatrix` クラスが動的に注入されます。
*   これにより、コードを変更することなく、CLI 引数だけでアーキテクチャの構成要素を変更可能です。

---

## 5. Directory Structure (Map)
