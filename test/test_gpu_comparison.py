#!/usr/bin/env python3
"""
GPU vs ハイブリッド実装の比較テスト
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gpu_only_eigenvals():
    """GPU単体での固有値計算テスト"""
    print("=== GPU単体での固有値計算テスト ===")
    
    key = random.PRNGKey(42)
    
    try:
        # GPUで行列生成
        W = random.uniform(key, (100, 100), minval=-1, maxval=1, dtype=jnp.float64)
        
        # GPUで固有値計算を試行
        start_time = time.time()
        eigenvals = jnp.linalg.eigvals(W)
        max_eigenval = jnp.max(jnp.abs(eigenvals))
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU固有値計算成功: {gpu_time:.6f}秒")
        print(f"   最大固有値: {max_eigenval:.6f}")
        return True, gpu_time
        
    except Exception as e:
        print(f"❌ GPU固有値計算失敗: {e}")
        return False, None

def test_hybrid_approach():
    """ハイブリッドアプローチテスト"""
    print("\n=== ハイブリッドアプローチテスト ===")
    
    key = random.PRNGKey(42)
    
    try:
        # CPUで初期化
        with jax.default_device(jax.devices('cpu')[0]):
            W_cpu = random.uniform(key, (100, 100), minval=-1, maxval=1, dtype=jnp.float64)
            
            # NumPyで固有値計算
            start_time = time.time()
            W_np = np.array(W_cpu)
            eigenvals = np.linalg.eigvals(W_np)
            max_eigenval = np.max(np.abs(eigenvals))
            cpu_time = time.time() - start_time
            
        # GPUに転送
        start_transfer = time.time()
        W_gpu = jax.device_put(jnp.array(W_np), jax.devices()[0])
        transfer_time = time.time() - start_transfer
        
        total_time = cpu_time + transfer_time
        
        print(f"✅ ハイブリッド計算成功: {total_time:.6f}秒")
        print(f"   CPU計算時間: {cpu_time:.6f}秒")
        print(f"   転送時間: {transfer_time:.6f}秒")
        print(f"   最大固有値: {max_eigenval:.6f}")
        
        return True, total_time
        
    except Exception as e:
        print(f"❌ ハイブリッド計算失敗: {e}")
        return False, None

def test_reservoir_performance():
    """実際のReservoir計算でのパフォーマンス比較"""
    print("\n=== Reservoir計算パフォーマンステスト ===")
    
    from reservoir import ReservoirComputer
    from reservoir.utils import generate_sine_data
    
    # テストデータ生成
    input_data, target_data = generate_sine_data(time_steps=1000)
    
    # Reservoir Computer初期化
    rc = ReservoirComputer(n_inputs=1, n_reservoir=100, n_outputs=1)
    
    # Reservoir実行時間測定
    start_time = time.time()
    states = rc.run_reservoir(input_data)
    reservoir_time = time.time() - start_time
    
    print(f"✅ Reservoir実行時間: {reservoir_time:.6f}秒")
    print(f"   使用デバイス: {jax.devices()[0]}")
    print(f"   状態形状: {states.shape}")
    
    return reservoir_time

def main():
    """メイン実行関数"""
    print("GPU vs ハイブリッド実装の比較テスト")
    print("=" * 50)
    print(f"JAXバージョン: {jax.__version__}")
    print(f"利用可能なデバイス: {jax.devices()}")
    print("=" * 50)
    
    # GPU単体テスト
    gpu_success, gpu_time = test_gpu_only_eigenvals()
    
    # ハイブリッドテスト
    hybrid_success, hybrid_time = test_hybrid_approach()
    
    # 実際のReservoir計算テスト
    reservoir_time = test_reservoir_performance()
    
    print("\n" + "=" * 50)
    print("📊 結果サマリー:")
    
    if gpu_success and hybrid_success:
        print(f"⚡ GPU固有値計算: {gpu_time:.6f}秒")
        print(f"🔄 ハイブリッド計算: {hybrid_time:.6f}秒")
        if gpu_time < hybrid_time:
            print("🏆 GPU単体が高速")
        else:
            print("🏆 ハイブリッドが同等またはより安定")
    elif hybrid_success:
        print("✅ ハイブリッドアプローチのみ成功")
        print("❌ GPU単体は環境問題で失敗")
        print("🎯 結論: ハイブリッドアプローチが必要")
    
    print(f"🚀 Reservoir計算: {reservoir_time:.6f}秒 (GPU実行)")

if __name__ == "__main__":
    main() 