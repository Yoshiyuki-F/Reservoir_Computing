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

```dash
uv add nvidia-cusparse-cu12==12.5.8.93 not the latest
```

**原因:** JAX 0.7.0+ はbundled CUDA librariesを使用するため

CUDA GPU is not recognized with the latest cusparse(12.5.9.5, 12.5.10.65) with jax or torch
RuntimeError: jaxlib/cuda/versions_helpers.cc:81: operation cusparseGetProperty(MAJOR_VERSION, &major) failed: The cuSPARSE library was not found. so 12.5.8.93