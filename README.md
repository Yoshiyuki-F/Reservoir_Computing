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

```bash
# 方法1: uv経由で実行（推奨）
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python examples/demo.py

# 方法2: 仮想環境アクティベート後に実行
source .venv/bin/activate
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda python examples/demo.py
```

このコマンドで以下の2つのデモンストレーションが実行されます：

1. **サイン波予測**: 複数の周波数を含むサイン波の時系列予測
2. **Lorenzアトラクター予測**: カオス時系列の予測

## ファイル構成

```
reservoir/
├── reservoir/                   # メインパッケージ
│   ├── __init__.py              # パッケージ初期化
│   ├── reservoir_computer.py    # メインのReservoirComputerクラス
│   ├── utils.py                # ユーティリティ関数とデータ生成
│   └── scripts.py              # Poetry実行スクリプト
├── tests/                      # テストファイル
│   ├── test_simple.py          # 基本機能テスト
│   ├── test_cuda.py            # GPU動作確認テスト
│   ├── test_gpu_comparison.py  # GPU vs ハイブリッド比較テスト
│   ├── test_eigenvalues_comparison.py  # 固有値計算比較テスト
│   └── test_edge_cases.py      # エッジケーステスト
├── examples/                   # サンプル・デモ
│   └── demo.py                 # メインデモンストレーション
├── scripts/                    # セットアップ・ユーティリティスクリプト
│   ├── install_cuda.sh         # GPU環境セットアップ
│   ├── run_gpu.sh              # GPU実行ラッパー
│   └── rebuild_test.sh         # 完全再構築テスト
├── outputs/                    # 生成ファイル
│   ├── sine_wave_prediction.png
│   └── lorenz_prediction.png
├── docs/                       # ドキュメント（将来用）
├── README.md                   # このファイル
└── pyproject.toml              # プロジェクト設定
```

## テスト実行

```bash
# uv環境でのテスト実行
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python tests/test_simple.py
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python tests/test_cuda.py
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python tests/test_gpu_comparison.py
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python tests/test_eigenvalues_comparison.py
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python tests/test_edge_cases.py
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


## GPU環境トラブルシューティング

### システムクラッシュ後の完全再構築手順

1. **基本環境の確認**
   ```bash
   # NVIDIA ドライバーの確認
   nvidia-smi
   
   # CUDA バージョンの確認
   nvcc --version
   ```

2. **プロジェクト環境の再構築**
   ```bash
   cd /path/to/reservoir
   
   # uv環境セットアップ
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   uv sync
   ```

### 一般的なGPU問題と解決策

#### 問題1: "cuSPARSE library was not found" エラー

**症状:**
```
RuntimeError: jaxlib/cuda/versions_helpers.cc:81: operation cusparseGetProperty(MAJOR_VERSION, &major) failed: The cuSPARSE library was not found.
```

**解決策:**
```bash
# uv環境での実行
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python your_script.py
```

**原因:** JAX 0.7.0+ はbundled CUDA librariesを使用するため、システムのLD_LIBRARY_PATH設定が競合を引き起こします。

#### 問題2: "Backend 'cuda' is not in the list of known backends" エラー

**症状:**
```
RuntimeError: Backend 'cuda' is not in the list of known backends: ['cpu', 'tpu'].
```

**解決策:**
```bash
# 環境変数をクリアして再実行
unset LD_LIBRARY_PATH
unset JAX_PLATFORMS
python your_script.py
```

#### 問題3: JAXがCPUにフォールバックする

**症状:**
```
WARNING: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

**解決策:**
```bash
# uv環境での再インストール
uv add "jax[cuda12]" --index https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 環境変数のクリア
unset LD_LIBRARY_PATH
```

### 推奨する実行方法

すべてのJAXスクリプトは以下の方法で実行してください：

```bash
# uv環境での実行
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python your_script.py

# または環境変数を分けて設定
export JAX_PLATFORMS=cuda
unset LD_LIBRARY_PATH
uv run python your_script.py
```

### GPU動作確認テスト

```bash
# uv環境での基本的なGPU動作確認
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python tests/test_cuda.py

# uv環境での詳細なGPU性能テスト
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python tests/test_gpu_comparison.py
```

### Linux Mint 特有の注意点

- **NVIDIA ドライバー:** Linux Mint では Driver Manager を使用してNVIDIA ドライバーをインストールすることを推奨
- **CUDA 互換性:** RTX 3060 では CUDA 12.x and driver 550.xx+ が必要
- **uv 環境:** システムの Python 環境との競合を避けるため uv の使用を強く推奨

### 簡単実行方法（推奨）

毎回環境変数を設定するのを避けるため、以下の方法を提供：

#### 1. ラッパースクリプト使用
```bash
# GPU環境で任意のPythonスクリプトを実行
./scripts/run_gpu.sh python examples/demo.py
./scripts/run_gpu.sh python tests/test_simple.py

# または直接実行
./scripts/run_gpu.sh examples/demo.py
```

#### 2. uv スクリプト使用（推奨）
```bash
# uv環境でGPUスクリプト実行
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run demo-gpu          # examples/demo.py をGPUで実行
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run test-gpu           # GPU動作テスト実行
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run reservoir-gpu examples/demo.py  # 任意のスクリプトをGPUで実行
```

#### 3. 従来の手動方式
```bash
# 毎回手動で設定する場合
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python examples/demo.py
```

**推奨:** 方法1または2を使用してください。システム全体への影響を避けられます。

### PyCharm IDE での設定

PyCharmの実行ボタン（▶️）でGPU環境を使用するための設定：

#### 1. Run Configuration の作成
1. **Run** → **Edit Configurations...** を開く
2. **+** → **Python** を選択
3. 以下のように設定：

**基本設定:**
- **Name:** `Reservoir GPU Demo` (任意)
- **Script path:** `/path/to/reservoir/examples/demo.py`
- **Python interpreter:** uv環境のPython (`/.venv/bin/python`)

**環境変数:**
- **Environment variables** 
  - `JAX_PLATFORMS=cuda`
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`


**重要:** `LD_LIBRARY_PATH`が設定されている場合は空に設定するか削除

#### 2. デフォルトConfiguration テンプレート設定
1. **Run** → **Edit Configurations...** 
2. **Templates** → **Python** を選択
3. **Environment variables** に以下を設定：
   - `JAX_PLATFORMS=cuda`
   - `XLA_PYTHON_CLIENT_PREALLOCATE=false`

これで新しいPython実行時に自動でGPU設定が適用されます。

#### 3. 実行確認
PyCharmの実行ボタン（▶️）で実行し、以下が表示されることを確認：
```
JAXバージョン: 0.7.0
利用可能なデバイス: [CudaDevice(id=0)]
```

#### 4. トラブルシューティング（PyCharm）
もしCPUで動作している場合：
1. **Terminal** タブで確認：
   ```bash
   echo $LD_LIBRARY_PATH
   ```
2. もし何か表示される場合、Run ConfigurationでLD_LIBRARY_PATHを空に設定
3. PyCharmを再起動

## 参考文献

- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
- Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training.
- [JAX CUDA Installation Guide](https://jax.readthedocs.io/en/latest/installation.html)
