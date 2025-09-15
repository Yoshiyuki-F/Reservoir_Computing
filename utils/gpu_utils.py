"""
Reservoir Computing用のGPUユーティリティ関数。
"""
from typing import Callable, Any

def check_gpu_available() -> bool:
    """JAXがGPUを認識し、利用可能かを確認。
    
    JAXが利用可能なデバイスリストを取得し、その中にGPUが含まれているかを
    検証します。簡単な計算テストも実行し、GPUが正常に動作することを
    確認します。
    
    Returns:
        GPUが利用可能な場合はTrue
    
    Raises:
        RuntimeError: GPUが検出されない、または計算テストに失敗した場合
        
    Examples:
        >>> try:
        ...     if check_gpu_available():
        ...         print("GPU is ready.")
        ... except RuntimeError as e:
        ...     print(e)
    """
    import jax
    import jax.numpy as jnp
    
    print("=== GPU認識確認 ===")

    try:
        devices = jax.devices()
        print(f"JAXバージョン: {jax.__version__}")

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
                "詳細なトラブルシューティングは docs/TROUBLESHOOTING.md を参照してください\n"
                "または --force-cpu オプションでCPU実行も可能です"
            )

        # 簡単なGPU計算テスト
        try:
            x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
            result = jnp.sum(x ** 2)

            # GPU名を取得（利用可能な場合）
            device = gpu_devices[0]
            device_name = "Unknown"
            try:
                # nvidia-mlまたはpynvmlを使ってGPU名を取得
                import subprocess
                nvidia_smi_output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                    universal_newlines=True
                ).strip().split('\n')[0]
                device_name = nvidia_smi_output
            except:
                # nvidia-smiが失敗した場合はデバイスIDのみ表示
                device_name = f"CUDA Device {device.id}"

            print(f"GPU detected: {device_name} (テスト成功: {float(result)})")
        except Exception as e:
            print(f"GPU計算テスト失敗: {e}")
            raise RuntimeError(f"GPU computation test failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"GPU確認エラー: {e}")
        raise RuntimeError(f"GPU availability check failed: {e}")


def require_gpu() -> Callable[..., Any]:
    """GPUを必須とするテスト用のデコレータ。
    
    このデコレータを付けたテスト関数は、実行前に`check_gpu_available`を
    呼び出します。GPUが利用できない場合、テストはスキップされず、
    エラーとして終了します。
    
    Returns:
        デコレートされたテスト関数ラッパー
        
    Examples:
        >>> @require_gpu()
        ... def test_gpu_specific_feature():
        ...     # GPUがなければこのテストは実行されない
        ...     assert True
    """
    import sys
    
    def decorator(test_func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                check_gpu_available()
                return test_func(*args, **kwargs)
            except RuntimeError as e:
                print(f"\nGPU REQUIREMENT FAILED: {e}")
                print("Exiting test due to GPU requirement...")
                sys.exit(1)
        return wrapper
    return decorator


def print_gpu_info() -> None:
    """現在JAXが使用しているGPUデバイス情報を表示。
    
    JAXが認識しているデバイスリストからGPUデバイスを特定し、
    その情報を標準出力に表示します。GPUが利用できない場合は
    その旨を伝えます。
    
    Examples:
        >>> print_gpu_info()
        Using GPU: T4
    """
    import jax
    
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        
        if gpu_devices:
            print(f"Using GPU: {gpu_devices[0]}")
        else:
            print("No GPU available - using CPU")
            
    except Exception as e:
        print(f"Cannot get GPU info: {e}")
