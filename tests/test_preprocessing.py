#!/usr/bin/env python3
"""
データ前処理ユーティリティのテスト
"""
import numpy as np
import jax.numpy as jnp
import pytest
from pipelines.preprocessing import normalize_data, denormalize_data


class TestNormalizeData:
    """normalize_data関数のテスト"""
    
    def test_basic_normalization(self):
        """基本的な正規化のテスト"""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized, mean, std = normalize_data(data)
        
        # 正規化後のデータが平均0、標準偏差1になっているかチェック
        assert abs(jnp.mean(normalized)) < 1e-10
        assert abs(jnp.std(normalized) - 1.0) < 1e-10
        
        # 返される平均と標準偏差が正しいかチェック
        assert abs(mean - 3.0) < 1e-10
        assert abs(std - np.sqrt(2.0)) < 1e-10
    
    def test_single_value(self):
        """単一値の正規化のテスト"""
        data = jnp.array([5.0])
        normalized, mean, std = normalize_data(data)
        
        # 単一値の場合、正規化後は0になる
        assert abs(normalized[0]) < 1e-10
        assert abs(mean - 5.0) < 1e-10
        # 標準偏差は最小値に設定される
        assert std >= 1e-12
    
    def test_constant_values(self):
        """定数値配列の正規化のテスト"""
        data = jnp.array([2.0, 2.0, 2.0, 2.0])
        normalized, mean, std = normalize_data(data)
        
        # 全て同じ値の場合、正規化後は全て0になる
        assert jnp.allclose(normalized, 0.0)
        assert abs(mean - 2.0) < 1e-10
        # 標準偏差は最小値に設定される
        assert std >= 1e-12
    
    def test_negative_values(self):
        """負の値を含む配列の正規化のテスト"""
        data = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        normalized, mean, std = normalize_data(data)
        
        # 正規化後のデータが平均0、標準偏差1になっているかチェック
        assert abs(jnp.mean(normalized)) < 1e-10
        assert abs(jnp.std(normalized) - 1.0) < 1e-10
        
        # 平均は0
        assert abs(mean) < 1e-10
    
    def test_2d_array(self):
        """2次元配列の正規化のテスト"""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized, mean, std = normalize_data(data)
        
        # 全体的な正規化が行われる
        assert abs(jnp.mean(normalized)) < 1e-10
        assert abs(jnp.std(normalized) - 1.0) < 1e-10
        
        # 元の配列と同じ形状を保持
        assert normalized.shape == data.shape


class TestDenormalizeData:
    """denormalize_data関数のテスト"""
    
    def test_basic_denormalization(self):
        """基本的な非正規化のテスト"""
        # 正規化されたデータ
        normalized = jnp.array([-1.414213, -0.707107, 0.0, 0.707107, 1.414213])
        mean = 3.0
        std = np.sqrt(2.0)
        
        denormalized = denormalize_data(normalized, mean, std)
        expected = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert jnp.allclose(denormalized, expected, atol=1e-5)
    
    def test_zero_normalized_data(self):
        """正規化データが全て0の場合のテスト"""
        normalized = jnp.array([0.0, 0.0, 0.0])
        mean = 5.0
        std = 2.0
        
        denormalized = denormalize_data(normalized, mean, std)
        expected = jnp.array([5.0, 5.0, 5.0])
        
        assert jnp.allclose(denormalized, expected)
    
    def test_single_value_denormalization(self):
        """単一値の非正規化のテスト"""
        normalized = jnp.array([1.5])
        mean = 10.0
        std = 3.0
        
        denormalized = denormalize_data(normalized, mean, std)
        expected = jnp.array([14.5])  # 1.5 * 3.0 + 10.0 = 14.5
        
        assert jnp.allclose(denormalized, expected)
    
    def test_negative_values_denormalization(self):
        """負の値を含む非正規化のテスト"""
        normalized = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        mean = 0.0
        std = 5.0
        
        denormalized = denormalize_data(normalized, mean, std)
        expected = jnp.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        
        assert jnp.allclose(denormalized, expected)
    
    def test_2d_array_denormalization(self):
        """2次元配列の非正規化のテスト"""
        normalized = jnp.array([[-1.0, 0.0], [0.0, 1.0]])
        mean = 5.0
        std = 2.0
        
        denormalized = denormalize_data(normalized, mean, std)
        expected = jnp.array([[3.0, 5.0], [5.0, 7.0]])
        
        assert jnp.allclose(denormalized, expected)
        assert denormalized.shape == normalized.shape


class TestNormalizationRoundTrip:
    """正規化と非正規化の往復テスト"""
    
    def test_roundtrip_consistency(self):
        """正規化→非正規化の一貫性テスト"""
        original = jnp.array([1.5, 2.7, 3.1, 4.9, 5.2, 6.8, 7.3])
        
        # 正規化
        normalized, mean, std = normalize_data(original)
        
        # 非正規化
        recovered = denormalize_data(normalized, mean, std)
        
        # 元のデータと復元されたデータが一致するかチェック
        assert jnp.allclose(original, recovered, rtol=1e-10)
    
    def test_roundtrip_2d_array(self):
        """2次元配列での往復テスト"""
        original = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        
        # 正規化
        normalized, mean, std = normalize_data(original)
        
        # 非正規化
        recovered = denormalize_data(normalized, mean, std)
        
        # 元のデータと復元されたデータが一致するかチェック
        assert jnp.allclose(original, recovered, rtol=1e-10)
    
    def test_roundtrip_with_large_values(self):
        """大きな値での往復テスト"""
        original = jnp.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        
        # 正規化
        normalized, mean, std = normalize_data(original)
        
        # 非正規化
        recovered = denormalize_data(normalized, mean, std)
        
        # 元のデータと復元されたデータが一致するかチェック
        assert jnp.allclose(original, recovered, rtol=1e-10)
    
    def test_roundtrip_with_small_values(self):
        """小さな値での往復テスト"""
        original = jnp.array([0.001, 0.002, 0.003, 0.004, 0.005])
        
        # 正規化
        normalized, mean, std = normalize_data(original)
        
        # 非正規化
        recovered = denormalize_data(normalized, mean, std)
        
        # 元のデータと復元されたデータが一致するかチェック
        assert jnp.allclose(original, recovered, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])