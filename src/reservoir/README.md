Revised Architecture Blueprint (V2.1)
1-6 Process Flow (Tensor Flow Style)
データの流れと各ステップの役割定義です。

STEP
1. Input Data - [Batch, Time, Features]
Examples: MNIST [28x28], Audio, Text

2. Preprocessing - [Batch, Time, Features] -> [Batch, Time, Features]
Role: Data scaling, polynomial features (Stateless or independent of model internal structure).
Examples: Raw, StandardScaler, DesignMatrix

3. Input Projection - [Batch, Time, Features] -> [Batch, Time, Hidden]
Role: Mapping input space to high-dimensional hidden space (Random or Learned).
Examples: Random Projection (W_in)
Note: FNN Student also uses this to match Teacher's input projection logic.

4. Adapter: Flatten is used here if Model requires 2D input [Batch, Time*Hidden] (only at FNN).

5. Model (Engine) - [Batch, Time, Hidden] -> [Batch, Time, Hidden] (Reservoir) OR [Batch, Hidden] (FNN)
Role: Stateful dynamics or Deep Non-linear mapping.
Examples: Classical Reservoir, Quantum Reservoir, FNN (Student)

6. Aggregation - [Batch, Time, Hidden] -> [Batch, Feature] (only at Reservoir).
Role: Temporal reduction to fixed feature vector. 
Examples: Last state, Mean, Concat

7. Readout - [Batch, Feature] -> [Batch, Output]
Role: Final decoding/classification.
Examples: Ridge Regression, Softmax
WHERE TO FIND THEM (Location Mapping)

Factory (/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/factory.py) should just include 4-5-6
where should /home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/generic_runner.py do then?

ファイル配置と責務のマッピングです。
data/ (Input Data) 1
layers/preprocessing.py (Preprocessing) 2
layers/projection.py (Input Projection) 3
layers/adapters.py (Structural Glues: Flatten, Reshape) 4
models/ (Model Engine & Assemblers) 5
    reservoir/, nn/, distillation/
layers/aggregation.py (Aggregation) 6
readout/ridge.py (Readout) 7



models/factory.py (Manufacturer)
責務: 4-6 (Engine Stack) の製造。
特徴: 状態を持たない。作って渡すだけ。

pipelines/generic_runner.py (Driver)
責務: 1-7 の実行（ChatGPTの言う「実験ロジックの正本」）。
特徴: 何のモデルか（FNNかReservoirか）を知らない。「学習して、特徴とって、Readoutする」という抽象的な手順だけを知っている。

pipelines/run.py (Manager/Frontend)
責務: 1-3 (Frontend) の準備 と、ドライバーへの指示。
特徴: 具体的なコンフィグ (ModelConfig) を解釈し、データを用意し、Factoryに製造を依頼し、Runnerに鍵を渡す。