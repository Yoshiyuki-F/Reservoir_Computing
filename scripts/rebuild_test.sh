#!/bin/bash
# システムクラッシュ後の完全再構築テストスクリプト

set -e  # エラー時に停止

echo "🔧 システムクラッシュ後の完全再構築テスト"
echo "============================================"

# 1. 基本環境確認
echo ""
echo "=== 1. 基本環境確認 ==="
echo "NVIDIA ドライバー確認:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits || { echo "❌ NVIDIA ドライバー未インストール"; exit 1; }

echo "CUDA バージョン確認:"
nvcc --version || { echo "❌ CUDA未インストール"; exit 1; }

# 2. Poetry環境確認
echo ""
echo "=== 2. Poetry環境確認 ==="
if ! command -v poetry &> /dev/null; then
    echo "⚠️ Poetry未インストール - 自動インストールを実行"
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

poetry --version || { echo "❌ Poetry インストール失敗"; exit 1; }

# 3. GPU環境セットアップテスト
echo ""
echo "=== 3. GPU環境セットアップテスト ==="
./scripts/install_cuda.sh || { echo "❌ GPU環境セットアップ失敗"; exit 1; }

# 4. GPU動作確認
echo ""
echo "=== 4. GPU動作確認テスト ==="
unset LD_LIBRARY_PATH
export JAX_PLATFORMS=cuda

echo "基本GPU動作テスト:"
python tests/test_cuda.py || { echo "❌ 基本GPU動作テスト失敗"; exit 1; }

echo ""
echo "GPU強制実行テスト:"
python -c "
import jax
import jax.numpy as jnp
devices = jax.devices()
if len(devices) == 0 or all(d.device_kind == 'cpu' for d in devices):
    print('❌ ERROR: GPU not detected!')
    exit(1)
print('✅ GPU detected:', devices)
x = jnp.array([1.0, 2.0, 3.0])
result = jnp.sum(x**2)
print('✅ GPU計算成功:', float(result))
" || { echo "❌ GPU強制実行テスト失敗"; exit 1; }

# 5. Reservoirライブラリテスト
echo ""
echo "=== 5. Reservoirライブラリテスト ==="
python -c "
from reservoir import ReservoirComputer
from reservoir.utils import generate_sine_data
import jax.numpy as jnp

# 小さなテストデータ
input_data, target_data = generate_sine_data(time_steps=100)
rc = ReservoirComputer(n_inputs=1, n_reservoir=50, n_outputs=1)
rc.train(input_data, target_data)
predictions = rc.predict(input_data)
print('✅ Reservoir Computing GPU動作確認成功')
print(f'   予測形状: {predictions.shape}')
print(f'   使用デバイス: {jnp.array(predictions).devices()}')
" || { echo "❌ Reservoirライブラリテスト失敗"; exit 1; }

echo ""
echo "🎉 完全再構築テスト成功！"
echo "システムクラッシュ後でもこのスクリプトで環境を復元できます。"