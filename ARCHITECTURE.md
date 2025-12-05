# 🏛️ ARCHITECTURE.md

**Project:** Reservoir Computing Framework (JAX/Flax)
**Version:** 2.0 (Stable / Refactored)
**Status:** Active Development
**Last Updated:** Nov 2025

---

## 1. Architectural Philosophy & Policy

本プロジェクトは、**「計算グラフの純粋性 (Pure Computation)」** と **「オブジェクト指向による構成管理 (Object-Oriented Configuration)」** を明確に分離する設計方針（**SAP**: Separation, Abstraction, Polytropism）を採用しています。

### 1.1 Core Principles (The Rules of Law)

1.  **Orchestrator Pattern (指揮者パターン)**
    *   **Class (`ReservoirModel`)** は「状態管理」「配管」「API提供」のみに責任を持ちます。
    *   **Function (`jax.lax.scan`)** は「実際の数値計算」のみに責任を持ちます。
    *   *Policy:* 計算ロジックを Class のメソッド内に隠蔽せず、可能な限り JAX の純粋関数として記述し、Class はそれを呼び出すラッパーとして機能させます。

2.  **Immutability of Physics (物理層の不変性)**
    *   リザバー（物理層）は、自分がどのタスク（MNISTやMackey-Glass）を解いているかを知ってはなりません。
    *   *Policy:* `ClassicalReservoir` などのノードクラスに、データセット固有の前処理やラベル変換ロジックを含めることは厳禁です。

3.  **Explicit Configuration (明示的な構成)**
    *   ハードコードされた定数（例: `N=100`, `alpha=1e-3`）をコードの深層に埋め込むことは禁止です。
    *   *Policy:* 全ての設定値は `config` 辞書または dataclass を通じて、CLI/Entrypoint から最深部のコンポーネントまで注入（Dependency Injection）されなければなりません。

4.  **Shape Consistency (テンソル形状の統一)**
    *   システム全体で以下の形状標準を厳守します：
        *   **Time-Series Input:** `(Batch, Time, Features)`
        *   **Reservoir State:** `(Batch, Time, Hidden)`
        *   **Readout Input:** `(Batch, Hidden)` (Flattened or Last-State) or `(Batch, Time, Hidden)`

---

## 2. System Overview

システムは **Component-based Architecture** です。データはパイプラインを通じて流れ、各ステージで変換されます。

flowchart LR
    Config[Configuration Dictionary] --> Orchestrator
    
    subgraph Pipeline [ReservoirModel Pipeline]
        Input((Input X)) --> Pre[TransformerSequence]
        Pre --> Node[Reservoir (Physics)]
        Node --> |States H| Strategy{Readout Strategy}
        Strategy --> |"Last / Mean / Flatten"| Features[Feature Vector]
        Features --> Readout[ReadoutModule (Ridge)]
        Readout --> Output((Output y))
    end
    
    subgraph Internals [JAX Core]
        Node -.-> |"jax.lax.scan"| Recurrence[Recurrent Dynamics]
    end


## 3. Component Details

### 3.1 The Orchestrator (`src/reservoir/models/reservoir/model.py`)
システムの中核です。`scikit-learn` ライクな `fit(X, y)` / `predict(X)` インターフェースを提供します。
*   **役割:**
    *   前処理 (`preprocess`) の適用。
    *   物理ノード (`reservoir`) の初期化と時間発展 (`forward`) の実行。
    *   時系列状態からの特徴量抽出 (`readout_mode`)。
    *   読み出し層 (`readout`) の学習と推論。
*   **Key Design:** `readout_mode="flatten"` をサポートし、短い時系列タスク（MNIST等）において全タイムステップの情報を活用可能にしています。

### 3.2 The Physics Layer (`src/reservoir/models/reservoir/`)
リザバーコンピューティングの「心臓部」です。
*   **`ClassicalReservoir`:**
    *   **Implementation:** `jax.lax.scan` を使用した高速なループ処理。Python の `for` ループは排除されています。
    *   **Interface:** `step(state, input)` および `forward(state, inputs)` を実装。
    *   **Artifacts:** 計算結果として `StepArtifacts(states=...)` を返し、内部状態の全履歴へのアクセスを提供します。

### 3.3 The Data Layer (`src/reservoir/data/`)
データのロードと生成を担当します。これまでのリファクタリングにより、責務が明確化されました。
*   **`registry.py`:** データセット名（文字列）とローダー関数のマッピングを一元管理。
*   **`loaders.py`:** 外部データ（MNIST等）の読み込みと整形。
*   **`generators.py`:** 合成データ（Mackey-Glass等）の数式生成。
*   **`config.py`:** データ生成パラメータの定義。

### 3.4 Preprocessing (`src/reservoir/components/preprocess/`)
入力データをリザバーに入れる前に変換します。
*   **`TransformerSequence`:** 複数の変換処理を連鎖させるコンテナ。
*   **`FeatureScaler`:** 正規化（Standardization/MinMax）。
*   **`DesignMatrix`:** 多項式特徴量拡張（Polynomial Expansion）。リザバーの非線形性を補完するために使用されます。

---

## 4. Key Implementation Patterns

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

開発者は以下の構造に従ってファイルを配置する必要があります。

src/reservoir/
├── components/          # Reusable building blocks
│   ├── preprocess/      # Scalers, DesignMatrix, TransformerSequence
│   ├── readout/         # RidgeRegression, LinearModels
│   └── utils/           # RNG helpers
├── core/                # Core Interfaces & Types (Abstract Base Classes)
│   ├── interfaces.py    # Protocol definitions (Transformer, Readout...)
│   └── ...
├── data/                # Data Access Layer
│   ├── registry.py      # Dataset Registry
│   ├── loaders.py       # MNIST, etc.
│   ├── generators.py    # Mackey-Glass, Sine Wave
│   └── config.py        # DataConfigs
├── models/              # High-level Models
│   ├── reservoir/model.py  # ReservoirModel (The Main Class)
│   ├── reservoir/       # Physics Implementations
│   │   ├── base.py      # Base Class
│   │   ├── classical.py # ESN / Echo State Network
│   │   └── (quantum*)   # Quantum Implementations (Future Work)
│   └── nn/              # Baseline Models (FNN, RNN)
└── utils/               # Generic Utilities (Metrics, GPU checks)


## 6. Extension Guidelines (For Agents)

新しい機能を追加する際は、以下の手順に従ってください。

### Scenario A: Adding a New Dataset
1.  **Loaderの実装:** `src/reservoir/data/loaders.py` に `load_my_dataset()` 関数を作成する。戻り値は `(X_train, y_train, X_test, y_test)` のJAX Arrayとする。
2.  **登録:** `src/reservoir/data/registry.py` に `@DatasetRegistry.register("my-dataset")` を追加する。
3.  **Config:** 必要に応じて `src/reservoir/data/config.py` にパラメータ定義を追加する。

### Scenario B: Adding a New Reservoir Type (e.g., Quantum)
1.  **継承:** `src/reservoir/models/reservoir/base.py` の `Reservoir` クラスを継承する。
2.  **実装:**
    *   `initialize_state(batch_size)`: 初期状態ゼロなどを返す。
    *   `step(state, input)`: 1ステップの時間発展記述。
    *   `forward(state, inputs)`: `jax.lax.scan` を用いて `step` を全系列に適用する。
    *   **重要:** 戻り値には必ず `StepArtifacts` を含めること。
3.  **統合:** `pipelines/run.py` の条件分岐に新しいタイプを追加し、Configからパラメータを渡す。

### Scenario C: Adding Preprocessing Logic
1.  **継承:** `src/reservoir/core/interfaces.py` の `Transformer` プロトコルに従う (`fit`, `transform`)。
2.  **配置:** `src/reservoir/components/preprocess/` に配置。
3.  **統合:** `pipelines/run.py` の `preprocess_steps` リスト構築ロジックに追加する。

---

## 7. Current Limitations & Future Work

*   **Quantum Implementations:** `QuantumAnalog` および `QuantumGateBased` は現在リファクタリング待ちの状態であり、V2アーキテクチャ（`scan`対応）に準拠していません。使用する際は `ClassicalReservoir` を参照実装として書き換える必要があります。
*   **Multi-layer Reservoirs:** 現在の Orchestrator は単層リザバーを想定しています。Deep ESN を実装する場合、`Reservoir` 抽象クラスをコンポジットパターンで拡張する必要があります。
