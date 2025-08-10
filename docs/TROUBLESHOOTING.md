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
# Poe the Poet使用（推奨）
uv run poe demo-sine-gpu

# または手動実行の場合
unset LD_LIBRARY_PATH && uv run python your_script.py
```

**原因:** JAX 0.7.0+ はbundled CUDA librariesを使用するため、システムのLD_LIBRARY_PATH設定が競合を引き起こします。

**解決法:** Poe the Poetタスクを使用すれば、pyproject.tomlの設定により`unset LD_LIBRARY_PATH`
とJAX_PLATFORMS=cudaが自動的に適用されます。

### 推奨する実行方法（Poe the Poet）

**デモ実行:**

```bash
# サイン波予測デモ
uv run poe demo-sine-gpu

# Lorenzアトラクター予測デモ  
uv run poe demo-lorenz-gpu
```

**テスト実行:**

```bash
# GPU基本動作確認
uv run poe test-gpu

# Reservoir Computing動作テスト
uv run poe test-simple-gpu

# GPU性能比較テスト
uv run poe test-gpu-comparison
```

**手動実行が必要な場合のみ:**

```bash
# カスタム設定での実行例
unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda uv run python -m reservoir --config your_config.json
```

### Linux Mint 特有の注意点

- **NVIDIA ドライバー:** Linux Mint では Driver Manager を使用してNVIDIA ドライバーをインストールすることを推奨
- **CUDA 互換性:** RTX 3060 では CUDA 12.x and driver 550.xx+ が必要
- **uv 環境:** システムの Python 環境との競合を避けるため uv の使用を強く推奨

### PyCharm IDE での設定

PyCharmでPoe the Poetタスクを実行する簡単な方法：

#### 方法1: uv Run Configuration （PyCharm 2025.2推奨）

1. **Run** → **Edit Configurations...** を開く
2. **+** → **uv** を選択
3. 以下のように設定：

**Poeタスク実行の場合:**
- **Name:** `Demo Main GPU`
- **Module name:** `poethepoet`
- **Parameters:** `demo-main-gpu`
- **Working directory:** `PROJECT_ROOT + /Resorvoir/reservoir`

**推奨:** PyCharm 2024.3.2以降ではuv Run Configurationが利用可能です。