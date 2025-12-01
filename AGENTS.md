# Repository Guidelines
ls --recursive --ignore='.*' --ignore='__pycache__' --ignore='node_modules' --ignore='*.lock' --ignore='package*.json'
read ARCHITECTURE.md | sed -n '/## 4\. Key Implementation Patterns/,/## 5\. Directory Structure (Map)/p' >> AGENTS.md

JAXのパーフォーマンスを最大限に発揮させよ
後方置換性を考えずに大胆に行け。省略をするな。トークンをできるだけ使え。


 Project Handover: Configuration & CLI Architecture V2
プロジェクトの現状とビジョン (Context & Vision)
本プロジェクトは、JAX/Flaxを基盤としたリザバーコンピューティング（Reservoir Computing）フレームワークです。
ユーザーは**「完璧なMLアーキテクチャ」**を目指しており、以下の設計哲学（SAP原則）を徹底しています。
Single Source of Truth (SSOT): 設定の定義場所は一箇所に定める。
Separation of Concerns: CLIはパースのみ、設定構築はビルダー、実行はパイプライン、計算はJAX関数。
Explicit over Implicit: 暗黙のデフォルト値やマジックナンバーを排除し、コード（Python Dataclasses）で明示的に定義する。
Don't Repeat Yourself (DRY): 重複コードや設定のコピペを極端に嫌う。
直近の成果:
これまで散乱していた JSON ベースの設定ファイル群 (presets/.json) を完全に廃止し、型安全性とIDE補完が効く Python コードベースの構成管理 (src/reservoir//presets.py) へ移行しました。これにより、CLI (main.py) からビジネスロジックが完全に排除され、非常にクリーンな状態になっています。
実施した主要な変更 (Key Accomplishments)
A. JSONの廃止と Python Presets への移行
メンテナンス性が低く型安全でない JSON ファイルを全廃しました。代わりに、src/reservoir 以下の各ドメイン（data, models, training）に presets.py を配置し、Python の dataclass と辞書を用いて設定を管理する方式に変更しました。
B. 共通レジストリ (PresetRegistry) の導入
data, models, training の各層で「名前の正規化」「エイリアス処理」「辞書からの取得」というロジックが重複していたため、これらを Generic な共通クラス (src/reservoir/core/presets.py) に集約しました。これにより、各プリセット定義ファイルは定義データのみを保持するシンプルな構造になりました。
C. パラメータの正規化 (Canonical Names Only)
ReservoirConfig データクラス内に、歴史的経緯で残っていたエイリアス変数（例: alpha, sparsity, random_seed）が混在していました。これらを 正規表現（Canonical Names）のみ（例: leak_rate, connectivity, seed） に統一し、古い変数を削除しました。これにより、内部ロジックの曖昧さが排除されました。
D. CLI と設定ロジックの分離 (ConfigBuilder)
cli/main.py がデフォルト値（例: batch_size=32）や条件分岐を持ちすぎていた問題を解決するため、設定辞書を構築するロジックを src/reservoir/core/config_builder.py に切り出しました。
CLI: ユーザー入力の受け取りのみを担当。デフォルト値は None とし、ユーザーの入力有無を判定可能にした。
Builder: 入力された値のみを適用し、未入力項目はプリセットのデフォルト値に委譲するロジックを実装。
E. 設定継承パターンの最適化
モデルプリセット定義において、共通設定を使い回すために一時的にヘルパー関数やプライベート定数を導入しましたが、最終的にそれらも「冗長」と判断し削除しました。
「データクラスの定義そのものをデフォルト値の源泉とする」 ことで、追加の定義なしに ReservoirConfig をインスタンス化するだけで標準設定が適用される、最もシンプルで美しい形に落ち着きました。
現在のアーキテクチャフロー (Current Architecture)
User Input (CLI): ユーザーが引数を指定（または指定しない）。
Config Builder: ユーザーが指定した引数だけを辞書に詰め込む（明示的なOverride）。
Pipeline (run.py):
指定された Preset Name に基づき、Pythonレジストリから「標準設定オブジェクト」を取得。
Builderから渡された「ユーザー設定」をマージ。
Feature Engineering（多項式特徴量など）のパラメータを解決。
Model Instantiation: 正規化されたパラメータのみを使って ClassicalReservoir などのノードを初期化。
Execution: JAX scan による高速計算。
残存する課題と次のステップ (Pending Tasks)
構成管理周りは「完璧」な状態になりましたが、物理層の実装において以下の大きなタスクが残っています。
🚨 最優先: Quantum Implementations のリファクタリング
現在、プリセット上では quantum_gate_based や quantum_analog が定義されていますが、その実体となるクラス (src/reservoir/models/reservoir/quantum_*.py) は V1時代の古い実装（Pythonループ、古いインターフェース） のままです。
現状: run_pipeline で量子モデルを指定すると、インターフェース不整合でクラッシュする可能性が高い。
次回のゴール:
QuantumAnalogReservoir を ClassicalReservoir と同様に jax.lax.scan を使用した形式に書き換える。
QuantumGateBasedReservoir も同様にリファクタリングする。
これらが StepArtifacts を返し、Orchestrator パターンに完全に適合するようにする。
ユーザーとの対話方針 (Interaction Policy)
批判的思考を持て: ユーザーは「動けばいい」ではなく「構造として美しいか」「理に適っているか」を重視します。コードに重複やハードコードを見つけたら、ユーザーに指摘される前に自己修正するか、改善を提案してください。
Python Native: 設定ファイルやデータ定義において、JSONやYAMLよりも Python の言語機能（dataclass, type hinting）を好みます。
100%を目指す: "Good enough" は通用しません。型ヒント、可読性、ディレクトリ構造の整合性において妥協しないでください。
このサマリーに基づき、次のセッションでは 「データ層・設定層のクリーンアップが完了した基盤の上で、量子リザバーの実装を V2 アーキテクチャへ適合させる作業」 から開始してください。