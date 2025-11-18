#!/usr/bin/env python3
"""
評価指標ユーティリティのテスト
"""
import jax.numpy as jnp
import pytest
from core_lib.utils import calculate_mse, calculate_mae


class TestCalculateMSE:
    """calculate_mse関数のテスト"""
    
    def test_perfect_prediction(self):
        """完全予測の場合のMSEテスト"""
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        mse = calculate_mse(predictions, targets)
        assert abs(mse - 0.0) < 1e-10
    
    def test_constant_error(self):
        """一定の誤差がある場合のMSEテスト"""
        predictions = jnp.array([2.0, 3.0, 4.0])
        targets = jnp.array([1.0, 2.0, 3.0])  # 全て1.0の差
        
        mse = calculate_mse(predictions, targets)
        expected_mse = 1.0  # (1^2 + 1^2 + 1^2) / 3 = 1.0
        assert abs(mse - expected_mse) < 1e-10
    
    def test_different_errors(self):
        """異なる誤差の場合のMSEテスト"""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 1.9, 3.2])
        # 誤差: [-0.1, 0.1, -0.2]
        # 二乗誤差: [0.01, 0.01, 0.04]
        
        mse = calculate_mse(predictions, targets)
        expected_mse = (0.01 + 0.01 + 0.04) / 3  # 0.02
        assert abs(mse - expected_mse) < 1e-10
    
    def test_negative_values(self):
        """負の値を含む場合のMSEテスト"""
        predictions = jnp.array([-1.0, 0.0, 1.0])
        targets = jnp.array([-1.5, 0.5, 1.5])
        # 誤差: [0.5, -0.5, -0.5]
        # 二乗誤差: [0.25, 0.25, 0.25]
        
        mse = calculate_mse(predictions, targets)
        expected_mse = 0.25
        assert abs(mse - expected_mse) < 1e-10
    
    def test_single_value(self):
        """単一値のMSEテスト"""
        predictions = jnp.array([5.0])
        targets = jnp.array([3.0])
        
        mse = calculate_mse(predictions, targets)
        expected_mse = (5.0 - 3.0) ** 2  # 4.0
        assert abs(mse - expected_mse) < 1e-10
    
    def test_2d_array(self):
        """2次元配列のMSEテスト"""
        predictions = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[1.1, 1.9], [2.8, 4.2]])
        # 誤差: [[-0.1, 0.1], [0.2, -0.2]]
        # 二乗誤差: [[0.01, 0.01], [0.04, 0.04]]
        
        mse = calculate_mse(predictions, targets)
        expected_mse = (0.01 + 0.01 + 0.04 + 0.04) / 4  # 0.025
        assert abs(mse - expected_mse) < 1e-10
    
    def test_large_values(self):
        """大きな値でのMSEテスト"""
        predictions = jnp.array([1000.0, 2000.0])
        targets = jnp.array([1010.0, 1990.0])
        # 誤差: [-10.0, 10.0]
        # 二乗誤差: [100.0, 100.0]
        
        mse = calculate_mse(predictions, targets)
        expected_mse = 100.0
        assert abs(mse - expected_mse) < 1e-10


class TestCalculateMAE:
    """calculate_mae関数のテスト"""
    
    def test_perfect_prediction(self):
        """完全予測の場合のMAEテスト"""
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        mae = calculate_mae(predictions, targets)
        assert abs(mae - 0.0) < 1e-10
    
    def test_constant_error(self):
        """一定の誤差がある場合のMAEテスト"""
        predictions = jnp.array([2.0, 3.0, 4.0])
        targets = jnp.array([1.0, 2.0, 3.0])  # 全て1.0の差
        
        mae = calculate_mae(predictions, targets)
        expected_mae = 1.0  # (1 + 1 + 1) / 3 = 1.0
        assert abs(mae - expected_mae) < 1e-10
    
    def test_different_errors(self):
        """異なる誤差の場合のMAEテスト"""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 1.9, 3.2])
        # 誤差: [-0.1, 0.1, -0.2]
        # 絶対誤差: [0.1, 0.1, 0.2]
        
        mae = calculate_mae(predictions, targets)
        expected_mae = (0.1 + 0.1 + 0.2) / 3  # 0.1333...
        assert abs(mae - expected_mae) < 1e-10
    
    def test_negative_values(self):
        """負の値を含む場合のMAEテスト"""
        predictions = jnp.array([-1.0, 0.0, 1.0])
        targets = jnp.array([-1.5, 0.5, 1.5])
        # 誤差: [0.5, -0.5, -0.5]
        # 絶対誤差: [0.5, 0.5, 0.5]
        
        mae = calculate_mae(predictions, targets)
        expected_mae = 0.5
        assert abs(mae - expected_mae) < 1e-10
    
    def test_single_value(self):
        """単一値のMAEテスト"""
        predictions = jnp.array([5.0])
        targets = jnp.array([3.0])
        
        mae = calculate_mae(predictions, targets)
        expected_mae = abs(5.0 - 3.0)  # 2.0
        assert abs(mae - expected_mae) < 1e-10
    
    def test_2d_array(self):
        """2次元配列のMAEテスト"""
        predictions = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[1.1, 1.9], [2.8, 4.2]])
        # 誤差: [[-0.1, 0.1], [0.2, -0.2]]
        # 絶対誤差: [[0.1, 0.1], [0.2, 0.2]]
        
        mae = calculate_mae(predictions, targets)
        expected_mae = (0.1 + 0.1 + 0.2 + 0.2) / 4  # 0.15
        assert abs(mae - expected_mae) < 1e-10
    
    def test_large_values(self):
        """大きな値でのMAEテスト"""
        predictions = jnp.array([1000.0, 2000.0])
        targets = jnp.array([1010.0, 1990.0])
        # 誤差: [-10.0, 10.0]
        # 絶対誤差: [10.0, 10.0]
        
        mae = calculate_mae(predictions, targets)
        expected_mae = 10.0
        assert abs(mae - expected_mae) < 1e-10


class TestMSEvsMAE:
    """MSEとMAEの比較テスト"""
    
    def test_outlier_sensitivity(self):
        """外れ値に対する感度の違いをテスト"""
        # 外れ値を含むデータ
        predictions = jnp.array([1.0, 2.0, 3.0, 10.0])  # 10.0が外れ値
        targets = jnp.array([1.1, 1.9, 3.1, 4.0])
        # 誤差: [-0.1, 0.1, -0.1, 6.0]
        
        mse = calculate_mse(predictions, targets)
        mae = calculate_mae(predictions, targets)
        
        # MSEは外れ値の影響で大きくなる（6^2=36が含まれる）
        expected_mse = (0.01 + 0.01 + 0.01 + 36.0) / 4  # 9.0075
        # MAEは外れ値の影響が小さい
        expected_mae = (0.1 + 0.1 + 0.1 + 6.0) / 4  # 1.575
        
        assert abs(mse - expected_mse) < 1e-10
        assert abs(mae - expected_mae) < 1e-10
        # MSEの方が外れ値の影響で大きくなる
        assert mse > mae
    
    def test_same_absolute_errors(self):
        """全ての誤差の絶対値が同じ場合のテスト"""
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.2, 1.8, 3.2, 3.8])
        # 誤差: [-0.2, 0.2, -0.2, 0.2]
        # 絶対値は全て0.2
        
        mse = calculate_mse(predictions, targets)
        mae = calculate_mae(predictions, targets)
        
        expected_mse = 0.2 ** 2  # 0.04
        expected_mae = 0.2
        
        assert abs(mse - expected_mse) < 1e-10
        assert abs(mae - expected_mae) < 1e-10
    
    def test_zero_errors(self):
        """誤差がゼロの場合、MSEとMAEが同じになることをテスト"""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        
        mse = calculate_mse(predictions, targets)
        mae = calculate_mae(predictions, targets)
        
        assert abs(mse - 0.0) < 1e-10
        assert abs(mae - 0.0) < 1e-10
        assert abs(mse - mae) < 1e-10


class TestEdgeCases:
    """エッジケースのテスト"""
    
    def test_very_small_errors(self):
        """非常に小さな誤差の場合のテスト"""
        predictions = jnp.array([1.000001, 2.000001, 3.000001])
        targets = jnp.array([1.0, 2.0, 3.0])
        
        mse = calculate_mse(predictions, targets)
        mae = calculate_mae(predictions, targets)
        
        expected_error = 1e-6
        expected_mse = expected_error ** 2
        expected_mae = expected_error
        
        assert abs(mse - expected_mse) < 1e-15
        assert abs(mae - expected_mae) < 1e-15
    
    def test_mixed_sign_errors(self):
        """正負混合の誤差の場合のテスト"""
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([0.8, 2.2, 2.9, 4.1])
        # 誤差: [0.2, -0.2, 0.1, -0.1]
        
        mse = calculate_mse(predictions, targets)
        mae = calculate_mae(predictions, targets)
        
        expected_mse = (0.2**2 + 0.2**2 + 0.1**2 + 0.1**2) / 4  # (0.04 + 0.04 + 0.01 + 0.01) / 4 = 0.025
        expected_mae = (0.2 + 0.2 + 0.1 + 0.1) / 4  # 0.15
        
        assert abs(mse - expected_mse) < 1e-10
        assert abs(mae - expected_mae) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
