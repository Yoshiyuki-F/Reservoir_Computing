"""GPU Test Utilities

GPU認識確認とエラーハンドリング用ユーティリティ
全てのテストで共通して使用する
"""

import sys
import jax
import jax.numpy as jnp


def check_gpu_available():
    """GPU認識確認とエラーハンドリング
    
    Returns:
        bool: GPU利用可能かどうか
    
    Raises:
        RuntimeError: GPU認識に失敗した場合
    """
    print("=== GPU認識確認 ===")
    
    try:
        devices = jax.devices()
        print(f"JAXバージョン: {jax.__version__}")
        print(f"利用可能なデバイス: {devices}")
        
        # GPU利用可能性チェック
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        
        if not gpu_devices:
            print("ERROR: GPU not detected!")
            print("Available devices:", devices)
            
            # CPU専用デバイスのチェック
            if devices and all('cpu' in str(d).lower() for d in devices):
                print("Only CPU devices detected - GPU initialization failed")
            
            raise RuntimeError(
                "GPU not detected. This is a GPU-required test.\n"
                "Troubleshoot:\n"
                "1. Check nvidia-smi works\n"
                "2. Verify JAX CUDA installation\n"
                "3. Ensure LD_LIBRARY_PATH is unset\n"
                "4. Set JAX_PLATFORMS=cuda"
            )
        
        print(f"GPU detected: {gpu_devices}")
        
        # 簡単なGPU計算テスト
        try:
            x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
            result = jnp.sum(x ** 2)
            print(f"GPU計算テスト成功: {float(result)}")
            print(f"計算デバイス: {x.devices()}")
        except Exception as e:
            print(f"GPU計算テスト失敗: {e}")
            raise RuntimeError(f"GPU computation test failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"GPU確認エラー: {e}")
        raise RuntimeError(f"GPU availability check failed: {e}")


def require_gpu():
    """GPUが必要なテスト用デコレータ関数
    
    GPU認識に失敗した場合、テストを終了する
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            try:
                check_gpu_available()
                return test_func(*args, **kwargs)
            except RuntimeError as e:
                print(f"\nGPU REQUIREMENT FAILED: {e}")
                print("Exiting test due to GPU requirement...")
                sys.exit(1)
        return wrapper
    return decorator


def print_gpu_info():
    """GPU情報を表示"""
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        
        if gpu_devices:
            print(f"Using GPU: {gpu_devices[0]}")
        else:
            print("No GPU available - using CPU")
            
    except Exception as e:
        print(f"Cannot get GPU info: {e}")