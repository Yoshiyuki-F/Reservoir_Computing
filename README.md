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

プロジェクトには以下の依存関係が必要です：

- Python 3.13+
- JAX (CUDA版) >= 0.6.1
- NumPy >= 2.2.6
- Matplotlib >= 3.10.3

## インストール

```bash
# プロジェクトディレクトリに移動
cd /path/to/reservoir

# 依存関係をインストール
pip install -e .
```

## 使用方法

### 基本的な使用例

```python
from reservoir import ReservoirComputer
from reservoir.utils import generate_sine_data, normalize_data

# データ生成
input_data, target_data = generate_sine_data(time_steps=1000)

# データ正規化
input_norm, input_mean, input_std = normalize_data(input_data)
target_norm, target_mean, target_std = normalize_data(target_data)

# Reservoir Computer初期化
rc = ReservoirComputer(
    n_inputs=1,
    n_reservoir=100,
    n_outputs=1,
    spectral_radius=0.95,
    input_scaling=1.0
)

# 訓練
rc.train(input_norm, target_norm)

# 予測
predictions = rc.predict(input_norm)
```

### デモンストレーション実行

```bash
python main.py
```

このコマンドで以下の2つのデモンストレーションが実行されます：

1. **サイン波予測**: 複数の周波数を含むサイン波の時系列予測
2. **Lorenzアトラクター予測**: カオス時系列の予測

## ファイル構成

```
reservoir/
├── reservoir/
│   ├── __init__.py              # パッケージ初期化
│   ├── reservoir_computer.py    # メインのReservoirComputerクラス
│   └── utils.py                # ユーティリティ関数とデータ生成
├── test/                       # テストファイル
│   ├── test_simple.py          # 基本機能テスト
│   ├── test_gpu_comparison.py  # GPU vs ハイブリッド比較テスト
│   ├── test_eigenvalues_comparison.py  # 固有値計算比較テスト
│   └── test_edge_cases.py      # エッジケーステスト
├── png/                        # 生成された画像ファイル
│   ├── sine_wave_prediction.png
│   └── lorenz_prediction.png
├── main.py                     # デモンストレーション
├── README.md                   # このファイル
└── pyproject.toml              # プロジェクト設定
```

## テスト実行

```bash
# 基本機能テスト
python test/test_simple.py

# GPU vs CPU 比較テスト
python test/test_gpu_comparison.py

# 固有値計算詳細比較
python test/test_eigenvalues_comparison.py

# エッジケーステスト
python test/test_edge_cases.py
```

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
4. **reg_param**: 過学習を防ぐ正則化パラメータ

## 例：カスタムデータでの使用

```python
import jax.numpy as jnp
from reservoir import ReservoirComputer

# カスタムデータの準備
# input_data: (time_steps, n_inputs)
# target_data: (time_steps, n_outputs)

rc = ReservoirComputer(
    n_inputs=input_data.shape[1],
    n_reservoir=200,
    n_outputs=target_data.shape[1],
    spectral_radius=0.9
)

rc.train(input_data, target_data)
predictions = rc.predict(input_data)
```

## 参考文献

- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
- Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training.
