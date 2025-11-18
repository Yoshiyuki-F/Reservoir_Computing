# Reservoir Computing with JAX

JAXを使った簡単なReservoir Computing実装です。

## 概要

Reservoir Computingは、固定されたランダムなリカレント層（reservoir）と訓練可能な出力層から構成されるニューラルネットワークです。reservoir層は固定されているため、出力層の重みのみを訓練すれば良く、計算効率的です。

## 特徴

- JAXによる高速数値計算
- カスタマイズ可能なreservoirパラメータ
- 複数のテストデータ生成機能
- 可視化機能付き

## 依存関係

pyproject.toml

## インストール

### uv環境セットアップ

```bash
# プロジェクトディレクトリに移動
cd /path/to/reservoir

# uvで依存関係インストール
uv sync

# 仮想環境のアクティベート（オプション）
source .venv/bin/activate
```

## 使用方法（Usage）

### CLI エントリポイント

このプロジェクトの推奨エントリポイントは `reservoir-cli` です（`pyproject.toml` の `[project.scripts]` で定義）。  
GPU環境では Poe タスク経由で実行するのがおすすめです。

```bash
# ヘルプ表示（CPU / 共通）
uv run reservoir-cli --help

# ヘルプ表示（GPU環境 + Poe）
uv run poe cli-gpu -- --help
```

### 典型的な実行例

#### 1. 古典的リザーバ（回帰）

```bash
# サイン波 + 古典的リザーバ（GPU, Poe 推奨）
uv run poe cli-gpu -- \
  sine \                  # (= sine_wave)
  cr \                    # (= classic_reservoir)
  600                     # (= --n-hiddenLayer 600)
```

uv run poe cli-gpu -- m fnn 30

#### 2. ゲート型量子リザーバ

```bash
uv run poe cli-gpu -- \
  lorenz \
  qr                      # (= gatebased_quantum)
```

#### 3. MNIST + FNN パイプライン

```bash
# 単純FNN
uv run poe cli-gpu -- \
  --dataset mnist \
  --model fnn_pretrained \
  --config presets/fnn_mnist_config.json

# FNN(b') バリアント
uv run poe cli-gpu -- \
  --dataset mnist \
  --model fnn_pretrained_b_dash \
  --config presets/fnn_b_dash_mnist_config.json
```

`--dataset` / `--model` は `src/core_lib/core/identifiers.py` の `Dataset` / `Pipeline` Enum で定義されている識別子の `value` と一致します。  
古典的リザーバではデフォルトで「Raw」前処理（生のリザーバ状態を使用）が有効になっています。  
スケーラ＋設計行列を使いたい場合は `--preprocessing default` を指定してください（`raw_standard` との切り替えは CLI が自動で行います）。

実行したコマンドライン自体が実験の完全な記録になるため、再現したい実験はそのままメモ・貼り付けしておくのがおすすめです。

## GPU要件

**このプロジェクトはデフォルトでGPUを必須としています。**

- GPU環境がない場合、ReservoirComputerは`RuntimeError`を発生させます
- CPUでの実行を強制する場合は、CLI で `--force-cpu` フラグを使用してください：

```bash
# CPU強制実行の例
uv run reservoir-cli \
  --dataset sine_wave \
  --model classic_reservoir \
  --n-hiddenLayer 600 \
  --force-cpu
```

Poe タスクを利用した GPU 実行の例は次の通りです：

```bash
# テスト（CPU）
uv run poe test

# GPU スモークテスト
uv run poe test-gpu

# 古典的リザーバ（GPU, Raw 前処理）
uv run poe cli-gpu -- sine cr 600
```

## ファイル構成

```
.
├── src/core_lib/               # メインPythonパッケージ
│   ├── cli.py                  # コマンドラインインターフェース
│   ├── core/                   # 設定・識別子・コンポーザ
│   ├── models/                 # 各種モデル定義
│   ├── reservoirs/             # リザーバ関連ロジック
│   └── utils/                  # GPU・前処理・メトリクス等ユーティリティ
├── presets/                   # 実験用プリセット（JSON設定群）
│   ├── models/
│   ├── training/
│   └── datasets/
├── docs/                       # ドキュメント
│   ├── TODO.md
│   └── TROUBLESHOOTING.md
├── outputs/                    # 生成ファイル
│   ├── lorenz_prediction.png
│   └── sine_wave_prediction.png
├── scripts/                    # ユーティリティスクリプト
│   ├── install_cuda.sh
│   ├── rebuild_test.sh
│   └── run_gpu.sh
├── tests/                      # テストファイル
│   ├── test_edge_cases.py
│   ├── test_eigenvalues_comparison.py
│   ├── test_gpu_comparison.py
│   └── test_reservoir_computer.py
├── pyproject.toml              # プロジェクト設定
└── README.md                   # このファイル
```

## モジュール構造詳細

### コアモジュール（`src/core_lib/` 配下）

#### `reservoir_computer.py`
- **ReservoirComputer**: メインのReservoir Computingクラス
- JAX JITコンパイルによる最適化されたCPU/GPU計算
- 統合されたbackend処理による簡潔な設計
- 各種リザーバー行列の生成・管理、状態更新、学習・予測機能を提供

#### `config.py`
- **ReservoirConfig**: リザーバーパラメータ設定クラス
- **DataGenerationConfig**: データ生成パラメータ設定クラス
- **TrainingConfig**: 学習パラメータ設定クラス
- **ExperimentConfig**: 実験全体の統合設定クラス
- **create_demo_config_template**: デモ設定テンプレート生成関数

### データ処理モジュール

#### `data.py`
- **generate_sine_data**: サイン波時系列データ生成
- **generate_lorenz_data**: Lorenzアトラクター時系列データ生成  
- **generate_mackey_glass_data**: Mackey-Glassカオス時系列データ生成

#### `preprocessing.py`
- **normalize_data**: データ正規化（0-1スケーリング）
- **denormalize_data**: 正規化解除

### ユーティリティモジュール（`src/core_lib/utils/`）

#### `gpu_utils.py`
- **check_gpu_available**: GPU利用可能性確認
- **require_gpu**: GPU必須環境の検証
- **print_gpu_info**: GPU情報表示

#### `metrics.py`
- **calculate_mse**: 平均二乗誤差計算
- **calculate_mae**: 平均絶対誤差計算

### パイプライン & 可視化モジュール（`pipelines/`）

#### `dynamic_runner.py`
- **run_dynamic_experiment / run_experiment**: 設定に基づき、データ生成、学習、評価、可視化までの一連の実験フローを実行するアプリケーション層のロジック。

#### `plotting.py`
- **plot_prediction_results**: 予測結果可視化（時系列グラフ生成）
- **plot_classification_results**: 分類タスクの混同行列・精度バーを可視化

### CLI エントリポイント（`cli/`）

#### `cli/main.py`
- **main**: `argparse`を用いてコマンドライン引数を解析し、`pipelines.dynamic_runner` を呼び出すアプリケーション層のエントリーポイント。

## ReservoirComputerクラスのパラメータ

- `n_inputs`: 入力次元数
- `n_hiddenLayer`: reservoir内のニューロン数
- `n_outputs`: 出力次元数
- `spectral_radius`: reservoirの固有値の最大絶対値（デフォルト: 0.95）
- `input_scaling`: 入力のスケーリング係数（デフォルト: 1.0）
- `noise_level`: reservoirに加えるノイズレベル（デフォルト: 0.001）
- `alpha`: leaky integrator parameter（デフォルト: 1.0）
- `random_seed`: 乱数シード（デフォルト: 42）

## 利用可能なデータ生成関数

### generate_sine_data()
複数の周波数を含むサイン波データを生成します。

### generate_lorenz_data()
Lorenzアトラクターの3次元カオス時系列データを生成します。

### generate_mackey_glass_data()
Mackey-Glassカオス時系列データを生成します。

## パフォーマンスチューニング

Reservoir Computingの性能は以下のパラメータで調整できます：

1. **reservoir_size**: 大きくすると表現力が向上しますが、計算コストが増加
2. **spectral_radius**: 0.9-1.0の範囲で調整。大きすぎると不安定になる
3. **input_scaling**: 入力データの特性に応じて調整
4. **ridge_lambdas**: 正則化強度のグリッド。自動スイープで最良のλを選択


## 参考文献

- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
- Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training.
- [JAX CUDA Installation Guide](https://jax.readthedocs.io/en/latest/installation.html)


TODO
ab wie viel bits kann es gut rechnen? 
literatur fastforward netz small mid large je nach ansatz literatur mnist(auf was literatur  beziehn? 
sequenzmnist  728px andere Reservoir ansat　(nie gefunden)
