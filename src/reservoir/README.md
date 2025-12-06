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

Factory should just include 4-5-6

ファイル配置と責務のマッピングです。
data/ (Input Data)
layers/preprocessing.py (Preprocessing)
layers/projection.py (Input Projection)
New: layers/adapters.py (Structural Glues: Flatten, Reshape)
models/ (Model Engine & Assemblers)
reservoir/, nn/, distillation/
layers/aggregation.py (Aggregation)
readout/ridge.py (Readout)