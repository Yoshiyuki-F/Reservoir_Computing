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
unset LD_LIBRARY_PATH uv run python your_script.py
```

**原因:** JAX 0.7.0+ はbundled CUDA librariesを使用するため、システムのLD_LIBRARY_PATH設定が競合を引き起こします。


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

#### 1. uv スクリプト使用（推奨）
```bash
# デモ実行（JAX_PLATFORMS=cudaは自動設定）
unset LD_LIBRARY_PATH && uv run demo-sine-gpu      # サイン波デモ
unset LD_LIBRARY_PATH && uv run demo-lorenz-gpu    # Lorenzデモ
unset LD_LIBRARY_PATH && uv run test-gpu           # GPU動作テスト実行
unset LD_LIBRARY_PATH && uv run test-simple-gpu    # 基本テスト実行
```

#### 2. ラッパースクリプト使用（従来）
```bash
# 複雑な環境セットアップが必要な場合
./scripts/run_gpu.sh python -m reservoir.cli --config configs/sine_wave_demo_config.json
```

#### 3. 従来の手動方式
```bash
# 毎回手動で設定する場合
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python -m reservoir.cli --config configs/sine_wave_demo_config.json
```

**推奨:** 方法1を使用してください。JAX_PLATFORMSが自動設定され、システム全体への影響を避けられます。

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