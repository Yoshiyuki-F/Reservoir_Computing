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

## 使用方法

### デモンストレーション実行

**推奨方法（Poe the Poet使用）:**
```bash
# all datas
uv run poe main
# サイン波デモ実行（環境変数は自動設定）
uv run poe sine

# Lorenzアトラクターデモ実行
uv run poe lorenz
```

**その他の実行方法:**
```bash
# 全てのテスト実行
uv run poe test-simple-gpu
uv run poe test-gpu-comparison

# カスタム設定での実行（環境変数なしの場合）
uv run python -m reservoir --config configs/sine_wave_demo_config.json --force-cpu
```

環境変数の詳細は docs/TROUBLESHOOTING.md を参照してください。

このコマンドで以下の2つのデモンストレーションが実行されます：

1. **サイン波予測**: 複数の周波数を含むサイン波の時系列予測
2. **Lorenzアトラクター予測**: カオス時系列の予測

## GPU要件

**このプロジェクトはデフォルトでGPUを必須としています。**

- GPU環境がない場合、ReservoirComputerは`RuntimeError`を発生させます
- CPUでの実行を強制する場合は、CLI で `--force-cpu` フラグを使用してください：

```bash
# CPU強制実行の例（Poeタスクがない場合のみ直接実行）
uv run python -m reservoir --config configs/sine_wave_demo_config.json --force-cpu
```

## ファイル構成

```
.
├── reservoir/                   # メインパッケージ
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                   # コマンドラインインターフェース
│   ├── config.py                # 設定クラス
│   ├── data.py                  # データ生成関数
│   ├── gpu_utils.py             # GPUユーティリティ
│   ├── metrics.py               # 評価指標
│   ├── plotting.py              # 可視化機能
│   ├── preprocessing.py         # データ前処理
│   ├── reservoir_computer.py    # ReservoirComputerクラス
│   └── runner.py                # 実験実行ロジック
├── configs/                    # 設定ファイル
│   ├── lorenz_demo_config.json
│   └── sine_wave_demo_config.json
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

### コアモジュール

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

### ユーティリティモジュール

#### `gpu_utils.py`
- **check_gpu_available**: GPU利用可能性確認
- **require_gpu**: GPU必須環境の検証
- **print_gpu_info**: GPU情報表示

#### `metrics.py`
- **calculate_mse**: 平均二乗誤差計算
- **calculate_mae**: 平均絶対誤差計算

#### `plotting.py`
- **plot_prediction_results**: 予測結果可視化（時系列グラフ生成）

### インターフェースと実行モジュール

#### `runner.py`
- **run_experiment**: 設定に基づき、データ生成、学習、評価、可視化までの一連の実験フローを実行するコアロジック。
- `cli.py`から呼び出されることで、コマンドラインと実行ロジックを分離。

#### `cli.py`
- **main**: `argparse`を用いてコマンドライン引数を解析し、`runner.py`の関数を呼び出す薄いラッパー。
- ユーザーが設定ファイルや実行オプションを簡単に指定できるようにする。

#### `__main__.py`
- `python -m reservoir`で実行された際のパッケージエントリーポイント。`cli.main()`を呼び出す。

#### `__init__.py`
- パッケージの初期化と、外部から利用される主要なクラスや関数をエクスポート。

## ReservoirComputerクラスのパラメータ

- `n_inputs`: 入力次元数
- `n_reservoir`: reservoir内のニューロン数
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
ab wie viel bits kann es gut rechnen? literatur fastforward netz small mid large je nach ansatz literatur mnist(auf was literatur  beziehn? sequenzmnist  728px andere Reservoir ansat