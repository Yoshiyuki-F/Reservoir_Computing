#!/usr/bin/env python3
"""
GPU環境での問題を再現するテスト
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# 強制的にfloat64を有効化
jax.config.update("jax_enable_x64", True)

def test_problematic_operations():
    """問題を起こしやすい演算のテスト"""
    print("=== 問題を起こしやすいGPU演算テスト ===")
    
    key = random.PRNGKey(42)
    
    # 大きな行列で複雑な演算
    print("1. 大きな行列での固有値計算...")
    try:
        W = random.uniform(key, (500, 500), minval=-1, maxval=1, dtype=jnp.float64)
        eigenvals = jnp.linalg.eigvals(W)
        print("    成功")
    except Exception as e:
        print(f"    失敗: {e}")
    
    # SVD分解
    print("2. SVD分解...")
    try:
        W = random.uniform(key, (200, 200), minval=-1, maxval=1, dtype=jnp.float64)
        U, s, Vh = jnp.linalg.svd(W)
        print("    成功")
    except Exception as e:
        print(f"    失敗: {e}")
    
    # Cholesky分解
    print("3. Cholesky分解...")
    try:
        A = random.uniform(key, (100, 100), dtype=jnp.float64)
        A = A @ A.T + 1e-6 * jnp.eye(100)  # 正定値行列にする
        L = jnp.linalg.cholesky(A)
        print("    成功")
    except Exception as e:
        print(f"    失敗: {e}")
    
    # QR分解
    print("4. QR分解...")
    try:
        W = random.uniform(key, (200, 200), dtype=jnp.float64)
        Q, R = jnp.linalg.qr(W)
        print("    成功")
    except Exception as e:
        print(f"    失敗: {e}")

def test_matrix_solve():
    """線形方程式求解のテスト"""
    print("\n=== 線形方程式求解テスト ===")
    
    key = random.PRNGKey(42)
    
    try:
        # Ridge回帰と同じような問題
        X = random.uniform(key, (1000, 200), dtype=jnp.float64)
        y = random.uniform(key, (1000, 1), dtype=jnp.float64)
        
        # 正規方程式
        XTX = X.T @ X
        XTy = X.T @ y
        
        # 正則化項追加
        reg_param = 1e-8
        A = XTX + reg_param * jnp.eye(XTX.shape[0], dtype=jnp.float64)
        
        # solve実行
        result = jnp.linalg.solve(A, XTy)
        print(" 線形方程式求解成功")
        print(f" 解の形状: {result.shape}")
        
    except Exception as e:
        print(f" 線形方程式求解失敗: {e}")

def simulate_reservoir_init():
    """実際のReservoir初期化をGPUで実行"""
    print("\n=== Reservoir初期化シミュレーション（GPU版）===")
    
    key = random.PRNGKey(42)
    n_reservoir = 200
    spectral_radius = 0.95
    
    try:
        # すべてGPUで実行
        key1, key2 = random.split(key, 2)
        
        # reservoir重み生成
        W_res = random.uniform(
            key2, 
            (n_reservoir, n_reservoir), 
            minval=-1, 
            maxval=1,
            dtype=jnp.float64
        )
        
        # スペクトル半径調整（GPU版）
        eigenvalues = jnp.linalg.eigvals(W_res)
        max_eigenvalue = jnp.max(jnp.abs(eigenvalues))
        max_eigenvalue = jnp.maximum(max_eigenvalue, 1e-8)
        W_res_scaled = (spectral_radius / max_eigenvalue) * W_res
        
        print(" GPU版Reservoir初期化成功")
        print(f"   最大固有値: {float(max_eigenvalue):.6f}")
        
        return True
        
    except Exception as e:
        print(f" GPU版Reservoir初期化失敗: {e}")
        return False

def main():
    print("GPU環境での問題を再現するテスト")
    print("=" * 50)
    print(f"JAXバージョン: {jax.__version__}")
    print(f"利用可能なデバイス: {jax.devices()}")
    print(f"float64有効: {jax.config.jax_enable_x64}")
    print("=" * 50)
    
    test_problematic_operations()
    test_matrix_solve()
    gpu_init_success = simulate_reservoir_init()
    
    print("\n" + "=" * 50)
    print("📊 結論:")
    if gpu_init_success:
        print(" この環境ではGPU版Reservoir初期化が可能")
        print(" ただし、ハイブリッドアプローチの方が:")
        print("   - より安定（環境依存性が少ない）")
        print("   - より高速（前回のテスト結果）")
        print("   - より互換性が高い")
    else:
        print(" この環境ではGPU版に問題があります")
        print(" ハイブリッドアプローチが必要です")

if __name__ == "__main__":
    main() 