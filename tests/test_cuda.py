#!/usr/bin/env python3
"""
CUDA GPU動作確認テスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.gpu_test_utils import require_gpu, print_gpu_info
import jax.numpy as jnp

@require_gpu()
def test_cuda_functionality():
    """CUDA GPU動作確認テスト"""
    print("=== CUDA GPU動作確認テスト ===")
    print_gpu_info()
    
    # 基本計算テスト
    x = jnp.array([1.0, 2.0, 3.0])
    result = jnp.sum(x**2) 
    print(f'GPU計算結果: {float(result)}')
    print(f'計算デバイス: {x.devices()}')
    
    # より複雑な計算テスト
    matrix = jnp.ones((1000, 1000))
    matrix_result = jnp.sum(matrix * matrix)
    print(f'行列計算結果: {float(matrix_result)}')
    
    print('CUDA GPU動作確認完了')

if __name__ == "__main__":
    try:
        test_cuda_functionality()
        print("\nすべてのCUDAテストが成功しました！")
    except Exception as e:
        print(f"CUDAテストエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)