#!/usr/bin/env python3
"""
固有値計算：GPU vs CPU 詳細比較テスト
"""

import jax
import jax.numpy as jnp
from jax import random, device_put
import numpy as np
import time

# float64を有効化
jax.config.update("jax_enable_x64", True)

def test_eigenvals_gpu_vs_cpu(matrix_sizes=[50, 100, 200, 500], num_trials=3):
    """異なるサイズの行列で固有値計算のGPU vs CPU比較"""
    print("🔍 固有値計算：GPU vs CPU 詳細比較")
    print("=" * 60)
    
    results = {}
    
    for size in matrix_sizes:
        print(f"\n 行列サイズ: {size}x{size}")
        print("-" * 40)
        
        gpu_times = []
        cpu_times = []
        transfer_times = []
        
        for trial in range(num_trials):
            key = random.PRNGKey(42 + trial)
            
            # === GPU版テスト ===
            try:
                # GPU上で行列生成
                W_gpu = random.uniform(key, (size, size), minval=-1, maxval=1, dtype=jnp.float64)
                
                # GPU固有値計算
                start_time = time.time()
                eigenvals_gpu = jnp.linalg.eigvals(W_gpu)
                max_eigenval_gpu = jnp.max(jnp.abs(eigenvals_gpu))
                # GPU計算結果を取得（同期）
                _ = float(max_eigenval_gpu)
                gpu_time = time.time() - start_time
                gpu_times.append(gpu_time)
                
                print(f"  Trial {trial+1} - GPU: {gpu_time:.6f}秒", end="")
                
            except Exception as e:
                print(f"  Trial {trial+1} - GPU: FAILED ({e})")
                gpu_times.append(float('inf'))
            
            # === CPU版テスト ===
            try:
                # CPU上で同じ行列生成
                with jax.default_device(jax.devices('cpu')[0]):
                    W_cpu = random.uniform(key, (size, size), minval=-1, maxval=1, dtype=jnp.float64)
                
                # NumPy固有値計算
                start_time = time.time()
                W_np = np.array(W_cpu)
                eigenvals_np = np.linalg.eigvals(W_np)
                max_eigenval_np = np.max(np.abs(eigenvals_np))
                cpu_time = time.time() - start_time
                cpu_times.append(cpu_time)
                
                # GPU転送時間測定
                start_transfer = time.time()
                _ = device_put(jnp.array(W_np), jax.devices()[0])
                transfer_time = time.time() - start_transfer
                transfer_times.append(transfer_time)
                
                print(f", CPU: {cpu_time:.6f}秒, 転送: {transfer_time:.6f}秒")
                
            except Exception as e:
                print(f", CPU: FAILED ({e})")
                cpu_times.append(float('inf'))
                transfer_times.append(0)
        
        # 統計計算
        if all(t != float('inf') for t in gpu_times):
            avg_gpu = np.mean(gpu_times)
            results[size] = {'gpu': avg_gpu}
        else:
            results[size] = {'gpu': None}
            
        if all(t != float('inf') for t in cpu_times):
            avg_cpu = np.mean(cpu_times)
            avg_transfer = np.mean(transfer_times)
            avg_total = avg_cpu + avg_transfer
            results[size].update({
                'cpu': avg_cpu,
                'transfer': avg_transfer,
                'total': avg_total
            })
        
        # 結果表示
        if results[size]['gpu'] and results[size].get('cpu'):
            speedup = results[size]['gpu'] / results[size]['total']
            if speedup > 1:
                print(f"   ハイブリッド(CPU+転送)が {speedup:.2f}x 高速")
            else:
                print(f"   GPUが {1/speedup:.2f}x 高速")
        elif results[size]['gpu']:
            print(f"   GPUのみ成功")
        elif results[size].get('cpu'):
            print(f"   CPUのみ成功")
    
    return results

def test_reservoir_specific_case():
    """Reservoir Computing特有のケースをテスト"""
    print("\n🧠 Reservoir Computing特有ケース")
    print("=" * 40)
    
    # Reservoir典型サイズ
    reservoir_sizes = [50, 100, 200, 500]
    spectral_radius = 0.95
    
    for size in reservoir_sizes:
        print(f"\nReservoir size: {size}")
        
        key = random.PRNGKey(42)
        
        # === GPU版 ===
        try:
            start_time = time.time()
            W_gpu = random.uniform(key, (size, size), minval=-1, maxval=1, dtype=jnp.float64)
            eigenvals = jnp.linalg.eigvals(W_gpu)
            max_eigenval = jnp.max(jnp.abs(eigenvals))
            W_scaled = (spectral_radius / max_eigenval) * W_gpu
            gpu_total = time.time() - start_time
            print(f"  GPU全体: {gpu_total:.6f}秒")
        except Exception as e:
            print(f"  GPU: FAILED - {e}")
            gpu_total = None
        
        # === ハイブリッド版 ===
        try:
            start_time = time.time()
            
            # CPU計算
            with jax.default_device(jax.devices('cpu')[0]):
                W_cpu = random.uniform(key, (size, size), minval=-1, maxval=1, dtype=jnp.float64)
            
            W_np = np.array(W_cpu)
            eigenvals = np.linalg.eigvals(W_np)
            max_eigenval = np.max(np.abs(eigenvals))
            W_scaled_np = (spectral_radius / max_eigenval) * W_np
            
            # GPU転送
            W_scaled_gpu = device_put(jnp.array(W_scaled_np), jax.devices()[0])
            
            hybrid_total = time.time() - start_time
            print(f"  ハイブリッド: {hybrid_total:.6f}秒")
            
            if gpu_total:
                ratio = gpu_total / hybrid_total
                print(f"  → ハイブリッドが {ratio:.2f}x 高速")
                
        except Exception as e:
            print(f"  ハイブリッド: FAILED - {e}")

def test_memory_usage():
    """メモリ使用量の比較"""
    print("\n メモリ使用量比較")
    print("=" * 30)
    
    size = 1000  # 大きな行列
    key = random.PRNGKey(42)
    
    print(f"行列サイズ: {size}x{size} ({size*size*8/1024/1024:.1f}MB)")
    
    # GPU版
    try:
        W_gpu = random.uniform(key, (size, size), dtype=jnp.float64)
        print(" GPU: メモリ確保成功")
        eigenvals = jnp.linalg.eigvals(W_gpu)
        print(" GPU: 固有値計算成功")
    except Exception as e:
        print(f" GPU: {e}")
    
    # CPU版
    try:
        with jax.default_device(jax.devices('cpu')[0]):
            W_cpu = random.uniform(key, (size, size), dtype=jnp.float64)
        print(" CPU: メモリ確保成功")
        
        W_np = np.array(W_cpu)
        eigenvals = np.linalg.eigvals(W_np)
        print(" CPU: 固有値計算成功")
    except Exception as e:
        print(f" CPU: {e}")

def main():
    print("🔍 固有値計算：GPU vs CPU 詳細分析")
    print("=" * 50)
    print(f"JAX: {jax.__version__}")
    print(f"デバイス: {jax.devices()}")
    print(f"float64: {jax.config.jax_enable_x64}")
    
    # 詳細比較
    results = test_eigenvals_gpu_vs_cpu()
    
    # Reservoir特有ケース
    test_reservoir_specific_case()
    
    # メモリ使用量
    test_memory_usage()
    
    print("\n" + "=" * 50)
    print(" 総合結論:")
    
    # 結果分析
    gpu_wins = 0
    hybrid_wins = 0
    
    for size, result in results.items():
        if result.get('gpu') and result.get('total'):
            if result['gpu'] < result['total']:
                gpu_wins += 1
            else:
                hybrid_wins += 1
    
    if hybrid_wins > gpu_wins:
        print(" ハイブリッドアプローチが優勢")
        print(" 推奨: 固有値計算もCPUで実行")
        print(" 理由:")
        print("   - より高速")
        print("   - より安定")
        print("   - GPU計算リソースを他の処理に節約")
    elif gpu_wins > hybrid_wins:
        print(" GPU固有値計算が優勢")
        print(" 推奨: 固有値計算もGPUで実行")
    else:
        print(" GPU・ハイブリッド互角")
        print(" 推奨: 安定性重視でハイブリッド")

if __name__ == "__main__":
    main() 