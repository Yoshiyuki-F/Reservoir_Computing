# Repository Guidelines
cd src/reservoir && ls --recursive
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


Read Architecure.md for detailed pipelines.