Revised Architecture Blueprint (V2.2)
1-8 Process Flow (Tensor Flow Style)
データの流れと各ステップの役割定義です。

STEP
1. Input Data - [N, Time, Features]
Role: Raw time-series input.
Examples: MNIST [54000, 28, 28], MackeyGlass [1, 16600, 1]

2. Preprocessing - [N, Time, Features] => [N, Time, Features]
Role: Stateless data scaling or polynomial features.
Examples: StandardScaler, MinMaxScaler. 

3. Input Projection(Feature Control) - [N, Time, Features] => [N, Time, Hidden]
Role: Mapping input space to a high-dimensional hidden space using a Matrix (e.g., $28 * 100$). 
This happens before the branching into Reservoir or FNN paths.
Examples: Random Projection (W_in), PCA, CropCenter, Resize

### 4-6. Branching Processes: Teacher vs. Student

Projected Data `[N, Time, Hidden]` の生成後（Step 3）、プロセスは2つのパスに分岐します。

#### Path A: Reservoir Process (The Teacher)
**役割:** リカレントな力学系（古典または量子）を用いて時間的な特徴を抽出し、教師信号となる特徴量（Aggregated Features）を生成します。

* **4A. Adapter (Identity/Passthrough)**
    * **Input:** `[N, Time, Hidden]`
    * **Action:** 何もしません（恒等写像）。Reservoirは時間次元を持つ3Dデータをそのまま受け取ります。

* **5A. Model (Reservoir Loop)**
    * **Input:** `[N, Time, Hidden]`
    * **Action:** 固定された（学習しない）リカレント結合を用いて、入力を高次元の状態空間へ展開します。
        * **Classical Reservoir:** $x_t = \tanh(W_{in}u_t + W_{res}x_{t-1})$ (ESN)
        * **Quantum Reservoir:** Quantum Circuit Evolution via TensorCircuit (JAX). Qubit states evolve via unitary gates + noise.
    * **Output:** **Reservoir States** `[N, Time, Hidden]` (Classical) or `[N, Time, Observables]` (Quantum)

* **6A. Aggregation (Creating Teacher Features)**
    * **Input:** `[N, Time, Hidden]` (Reservoir States)
    * **Action:** 時間次元を集約または整形し、最終的な特徴ベクトルを生成します。
        * **Classification (MNIST):** **Mean Pooling**. 時間方向に平均化します。
            * Shape: `[N, 28, 100]` $\rightarrow$ `[N, 100]`
        * **Regression (MackeyGlass):** **Sequence/Identity**. バッチ次元の除去やシーケンスの整列を行います。
            * Shape: `[1, 16600, 100]` $\rightarrow$ `[16600, 100]`
    * **Output:** **Aggregated Features** `[N', Hidden]` Always 2D output!

---

#### Path B: FNN Distillation Process (The Student)
**役割:** 教師（Path A）が生成した特徴量「Aggregated Features」を、フィードフォワード・ニューラルネットワーク（FNN）で模倣するように学習します。

* **4B. Adapter (Reshaping for FNN)**
    * **Input:** `[N, Time, Hidden]` (Projected Data)
    * **Action:** 3DデータをFNNが処理可能な2D形式に変換します。
        * **Classification (MNIST):** **Flatten**. 時間と特徴量を結合します。
            * Shape: `[N, Time \times Hidden]` (例: `[N, 2800]`)
        * **Regression (MackeyGlass):** **Windowing**. スライディングウィンドウを作成します。
            * Shape: `[Time - Window + 1', Window \times Hidden]` (例: `[16598, 300]`)

* **5B. Model (FNN Engine & Distillation)**
    * **Input:** `[N', InputDim_FNN]` (Flattened or Windowed Data)
    * **Action:** 多層パーセプトロン（例: 64x32層）を通して特徴抽出を行います。
    * **Distillation (蒸留):** このFNNの重み $W_{fnn}$ は、出力が **Step 6Aの「Aggregated Features」に近づくように** 学習（Train）されます。
        * *Goal:* $FNN(x) \approx \text{Aggregation}(\text{Reservoir}(x))$
    * **Output:** **FNN Features** `[N', Hidden]`
        * ※ Path Bには明示的なStep 6（Aggregation）はなく、FNNの出力がそのまま最終特徴量となります。


7. Readout - [N', Hidden] => [N', Output]
Role: Final decoding/classification using linear models. Both paths use their own Readout.
7A (Reservoir): Ridge Regression trained on Aggregated Features.
7B (FNN): Ridge Regression trained on FNN Features.
Targets Y: One-hot encoding [N, 10] or continuous values.


ファイル配置と責務のマッピングです。
data/ (Input Data) 1
layers/preprocessing.py (Preprocessing) 2
layers/projection.py (Input Projection) 3
layers/adapters.py (Structural Glues: Flatten, Reshape) 4
models/ (Model Engine & Assemblers) 5
    reservoir/
        classical.py (ESN)
        quantum/ (Quantum Reservoir: TensorCircuit + JAX)
            backend.py (Circuit Execution)
            functional.py (Gate Logic, Noise, JIT Scan)
            model.py (State Management)
    nn/ (FNN)
    distillation/ (Distillation Logic)
layers/aggregation.py (Aggregation) 6
readout/ridge.py (Readout) 7



models/factory.py (Manufacturer)
責務: 4-6 (Engine Stack) の製造。
特徴: 状態を持たない。作って渡すだけ。

pipelines/components/executor.py (Driver)
責務: 1-7 の実行（実験ロジックの正本）。
特徴: 何のモデルか（FNN/Classical/Quantum）を知らない。「学習して、特徴とって、Readoutする」という抽象的な手順だけを知っている。

pipelines/run.py (Manager/Frontend)
責務: 1-3 (Frontend) の準備 と、ドライバーへの指示。
特徴: 具体的なコンフィグ (RunConfig) を解釈し、データを用意し、Factoryに製造を依頼し、Runnerに鍵を渡す。