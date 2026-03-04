# Architecture Refactoring Plan: Achieving SOLID & Clean Architecture

## 1. 現状の課題と目的 (The Problem & Goal)

現在のパイプラインは、`FNN_Distillation` などのモデルが追加されたことで、「データの一括ロード（NumPy配列）」と「GPUメモリの制約（OOM）」が衝突しています。これを回避するために、`batched_compute` を用いたバッチ処理や、Projection の合成（Fused）ロジックが `DistillationModel` や `PipelineExecutor` の中に漏れ出し、**SRP（単一責任の原則）** や **LSP（リスコフの置換原則）** に違反しています。

本リファクタリングの目的は、`DataCoordinator` を「真のデータストリーミング層（DataLoader）」に昇格させ、モデルやExecutorからバッチ化やProjection合成のインフラ的責務を排除することです。これにより、Clean Architecture と SOLID 原則を遵守した設計を実現します。

---

## 2. 変更するコンポーネントと役割

### Phase 1: データ層の抽象化 (DataCoordinator の強化)
現在、単に巨大な NumPy 配列を返すだけになっている `DataCoordinator` を、バッチ単位でデータを供給する **Iterator（Generator）プロバイダ** に作り直します。

**変更点: `src/reservoir/pipelines/components/data_coordinator.py`**
*   **追加:** `get_train_batches(batch_size: int, projection: Projection | None = None) -> Iterator[tuple[JaxF64, JaxF64]]`
    *   内部で生の `train_X`, `train_y` をスライスし、もし `projection` が渡されていればその場で適用し、`JaxF64` に変換して yield する。
*   **追加:** `get_eval_batches(...)` (Validation / Test 用)
*   **利点:** 巨大なデータがメモリに展開されることを防ぎ、Projection の合成処理（Fused）を Coordinator の責務として完全に隠蔽します。

### Phase 2: モデル層からバッチ処理ロジックの排除 (SRPの回復)
`DataCoordinator` がバッチジェネレータを提供できるようになったことで、モデル側で `batched_compute` を呼び出す必要がなくなります。

**変更点: `src/reservoir/models/distillation/model.py`**
*   **削除:** `_compute_teacher_targets_batched` メソッド。
*   **修正:** `train` メソッド。
    *   全件を一括で処理して `teacher_targets` を作るのではなく、`DataCoordinator` から渡されたバッチジェネレータを受け取り、ループの中で `teacher` に推論させてターゲットを作り、そのまま `student` の学習ステップに渡す。
    *   引数から `projection_layer` のハック（`kwargs`への依存）を削除。

**変更点: `src/reservoir/models/nn/base.py` (BaseFlaxModel)**
*   **修正:** `train` メソッド。
    *   入力として `inputs: JaxF64` (全データ) ではなく、DataLoader（Generator）を受け取るように変更。
    *   内部の epoch ループで、Generator からバッチを取り出して `train_step_jit` を呼ぶシンプルな設計にする。

### Phase 3: Executor の責務純化 (LSPの回復)
`PipelineExecutor` から、モデルごとの分岐や `batched_compute` を用いた Fused ロジックを削除し、「Coordinator から Data をもらい、Model に渡し、Readout を呼ぶ」という本来のオーケストレーションのみに専念させます。

**変更点: `src/reservoir/pipelines/components/executor.py`**
*   **修正:** `run` メソッド。
    *   `train_logs = self.stack.model.train(coordinator.get_train_batches(...))` のように呼び出す。
*   **修正:** `_extract_all_features` (および `_compute_split`)。
    *   `batched_compute` への直接依存を排除。
    *   `coordinator.get_eval_batches(...)` からバッチを受け取り、`model.predict` などを呼び出して結果をリストに蓄積 $\rightarrow$ 最後に結合（`np.concatenate`）して返すシンプルなロジックにする。

### Phase 4: `batched_compute.py` の廃止または縮小
バッチ処理のオーケストレーションが `DataCoordinator` とモデルの学習ループに自然な形で組み込まれるため、汎用的な `batched_compute` ユーティリティは（少なくとも現在の形の巨大なラッパーとしては）不要になります。
*   **削除・整理:** `src/reservoir/utils/batched_compute.py` を廃止するか、単なるヘルパー関数レベルまで縮小する。

---

## 3. 実装のステップ (Action Plan)

1.  **[Preparation]** `core/types.py` に DataLoader や Batch Iterator に対応する型定義を追加する。
2.  **[Step 1]** `data_coordinator.py` を改修し、Projection と JAX 変換を内包した Generator メソッドを実装する。
3.  **[Step 2]** `models/nn/base.py` の学習ループを、Generator を消費する形に書き換える。
4.  **[Step 3]** `models/distillation/model.py` を改修し、Teacher のターゲット生成と Student の学習をバッチレベルで統合する（OOM対策コードの削除）。
5.  **[Step 4]** `executor.py` を改修し、データの受け渡し方を Generator ベースに変更する。
6.  **[Validation]** `uv run python -m reservoir.cli.main --model fnn_distillation_classical --dataset mnist` を実行し、OOM が発生せず、クリーンなログが出力されることを確認する。
7.  **[Cleanup]** `batched_compute.py` の整理と、`ARCHITECTURE.md` の更新。

この設計変更により、「モデルはデータの出所や形状変換を知らなくてよい」「Executorはメモリ管理を知らなくてよい」という本来の Clean Architecture が達成されます。