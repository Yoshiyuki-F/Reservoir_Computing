#!/bin/bash
# CUDA環境セットアップスクリプト - GPU専用動作
# 新しいvenv環境でCUDA対応JAXを再インストールするためのスクリプト
# システムクラッシュ後の再構築に対応

echo "CUDA GPU専用環境のセットアップを開始します..."

# IMPORTANT: JAX 0.7.0+ はbundled CUDA librariesを使用するため
# LD_LIBRARY_PATH設定は競合を引き起こします
# システム確認時のみ一時的に設定し、JAX実行時はunsetします

# 一時的な環境変数設定（システム確認用）
export CUDA_HOME=/usr/local/cuda
export TEMP_LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:$PATH"

# 環境確認
echo "=== システム確認 ==="
echo "CUDA Version:"
nvcc --version || { echo "❌ CUDA not found"; exit 1; }
echo "GPU Status:"
nvidia-smi || { echo "❌ NVIDIA GPU not found"; exit 1; }

# cuSPARSE確認（一時的にLD_LIBRARY_PATH設定）
echo ""
echo "=== cuSPARSE確認 ==="
export LD_LIBRARY_PATH="$TEMP_LD_LIBRARY_PATH"
ldconfig -p | grep cusparse || { echo "❌ cuSPARSE not found"; exit 1; }
echo "✅ cuSPARSE found"

# Poetry経由での依存関係インストール
echo ""
echo "=== Poetry依存関係インストール ==="
poetry install || { echo "❌ Poetry install failed"; exit 1; }

# CRITICAL: LD_LIBRARY_PATHをunsetしてJAXテスト実行
echo ""
echo "=== GPU専用動作確認（LD_LIBRARY_PATH conflict回避） ==="
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

echo "環境変数状態："
echo "  CUDA_HOME: $CUDA_HOME"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-'(unset)'}"
echo "  JAX_PLATFORMS: $JAX_PLATFORMS"

python -c "
import os
import jax
import jax.numpy as jnp

print('JAX Version:', jax.__version__)
print('Available devices:', jax.devices())

# GPU専用チェック
devices = jax.devices()
if len(devices) == 0 or all(d.device_kind == 'cpu' for d in devices):
    print('❌ ERROR: GPU not detected! Only CPU available.')
    print('TROUBLESHOOT: LD_LIBRARY_PATH conflict - JAX falling back to CPU')
    exit(1)

gpu_devices = [d for d in devices if d.device_kind == 'gpu']
if not gpu_devices:
    print('❌ ERROR: No GPU devices found!')
    exit(1)

print('✅ GPU detected:', gpu_devices)
print('Default device:', devices[0])

# GPU計算テスト
try:
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = jnp.sum(x**2)
    print('✅ GPU計算テスト成功:', float(result))
    print('計算デバイス:', x.devices())
except Exception as e:
    print('❌ GPU計算テスト失敗:', e)
    exit(1)
"

echo ""
echo "✅ CUDA GPU専用環境セットアップ完了！"
echo ""
echo "🔧 重要な使用上の注意："
echo "   JAXを実行する際は必ず以下を実行してください："
echo "   export JAX_PLATFORMS=cuda"
echo "   unset LD_LIBRARY_PATH"
echo ""
echo "   または一行で："
echo "   unset LD_LIBRARY_PATH && JAX_PLATFORMS=cuda python your_script.py"
echo ""
echo "📖 詳細なトラブルシューティングはREADME.mdを参照してください。"