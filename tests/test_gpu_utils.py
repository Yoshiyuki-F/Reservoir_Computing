#!/usr/bin/env python3
"""
GPU ユーティリティ関数のテスト

注意：このテストはGPU環境でのみ実行され、CPU環境では失敗します。
GPU環境が利用できない場合は、テストをスキップしてください。
"""
import pytest
from unittest.mock import patch, MagicMock
import sys
from io import StringIO
from utils.gpu_utils import check_gpu_available, require_gpu, print_gpu_info


class TestCheckGPUAvailable:
    """check_gpu_available関数のテスト"""
    
    @patch('jax.devices')
    @patch('jax.numpy.array')
    @patch('jax.numpy.sum')
    @patch('jax.__version__', '0.7.0')
    def test_gpu_available_success(self, mock_sum, mock_array, mock_devices):
        """GPU利用可能な場合の成功テスト"""
        # GPUデバイスをモック
        mock_gpu_device = MagicMock()
        mock_gpu_device.__str__.return_value = "CUDA device 0"
        mock_devices.return_value = [mock_gpu_device]
        
        # JAX配列と計算をモック
        mock_jax_array = MagicMock()
        mock_jax_array.devices.return_value = [mock_gpu_device]
        mock_array.return_value = mock_jax_array
        mock_sum.return_value = 55.0  # 1^2 + 2^2 + ... + 5^2 = 55
        
        # 標準出力をキャプチャ
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = check_gpu_available()
        
        assert result is True
        output = captured_output.getvalue()
        assert "=== GPU認識確認 ===" in output
        assert "GPU detected:" in output
        assert "GPU計算テスト成功: 55.0" in output
    
    @patch('jax.devices')
    def test_no_gpu_devices(self, mock_devices):
        """GPU デバイスが見つからない場合のテスト"""
        # CPUデバイスのみをモック
        mock_cpu_device = MagicMock()
        mock_cpu_device.__str__.return_value = "CPU device 0"
        mock_devices.return_value = [mock_cpu_device]
        
        # 標準出力をキャプチャ
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            with pytest.raises(RuntimeError, match="GPU not detected"):
                check_gpu_available()
        
        output = captured_output.getvalue()
        assert "ERROR: GPU not detected!" in output
        assert "Only CPU devices detected" in output
    
    @patch('jax.devices')
    def test_empty_devices_list(self, mock_devices):
        """デバイスリストが空の場合のテスト"""
        mock_devices.return_value = []
        
        with pytest.raises(RuntimeError, match="GPU not detected"):
            check_gpu_available()
    
    @patch('jax.devices')
    @patch('jax.numpy.array')
    def test_gpu_computation_failure(self, mock_array, mock_devices):
        """GPU計算テストが失敗する場合のテスト"""
        # GPUデバイスはあるがJAX計算で例外が発生
        mock_gpu_device = MagicMock()
        mock_gpu_device.__str__.return_value = "CUDA device 0"
        mock_devices.return_value = [mock_gpu_device]
        
        # JAX配列作成で例外を発生させる
        mock_array.side_effect = RuntimeError("CUDA initialization failed")
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            with pytest.raises(RuntimeError, match="GPU computation test failed"):
                check_gpu_available()
        
        output = captured_output.getvalue()
        assert "GPU計算テスト失敗:" in output


class TestRequireGPU:
    """require_gpu デコレータのテスト"""
    
    @patch('utils.gpu_utils.check_gpu_available')
    def test_gpu_available_decorator_success(self, mock_check_gpu):
        """GPU利用可能時のデコレータ成功テスト"""
        # check_gpu_available が成功すると仮定
        mock_check_gpu.return_value = True
        
        # デコレータを適用
        @require_gpu()
        def test_function():
            return "GPU test passed"
        
        # 関数が正常に実行される
        result = test_function()
        assert result == "GPU test passed"
        mock_check_gpu.assert_called_once()
    
    @patch('utils.gpu_utils.check_gpu_available')
    @patch('sys.exit')
    def test_gpu_unavailable_decorator_exit(self, mock_exit, mock_check_gpu):
        """GPU利用不可時のデコレータ終了テスト"""
        # check_gpu_available が失敗すると仮定
        mock_check_gpu.side_effect = RuntimeError("GPU not found")
        
        @require_gpu()
        def test_function():
            return "Should not reach here"
        
        # 標準出力をキャプチャ
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            test_function()
        
        # sys.exit が呼ばれることを確認
        mock_exit.assert_called_once_with(1)
        
        output = captured_output.getvalue()
        assert "GPU REQUIREMENT FAILED:" in output
        assert "Exiting test due to GPU requirement..." in output
    
    @patch('utils.gpu_utils.check_gpu_available')
    def test_decorated_function_with_args(self, mock_check_gpu):
        """引数付き関数のデコレータテスト"""
        mock_check_gpu.return_value = True
        
        @require_gpu()
        def test_function_with_args(x, y, z=None):
            return f"x={x}, y={y}, z={z}"
        
        result = test_function_with_args(1, 2, z=3)
        assert result == "x=1, y=2, z=3"
        mock_check_gpu.assert_called_once()


class TestPrintGPUInfo:
    """print_gpu_info関数のテスト"""
    
    @patch('jax.devices')
    def test_gpu_info_with_gpu(self, mock_devices):
        """GPU存在時の情報表示テスト"""
        mock_gpu_device = MagicMock()
        mock_gpu_device.__str__.return_value = "CUDA device 0 (Tesla T4)"
        mock_devices.return_value = [mock_gpu_device]
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            print_gpu_info()
        
        output = captured_output.getvalue()
        assert "Using GPU:" in output
        assert "Tesla T4" in output or "CUDA device 0" in output
    
    @patch('jax.devices')
    def test_gpu_info_no_gpu(self, mock_devices):
        """GPU非存在時の情報表示テスト"""
        mock_cpu_device = MagicMock()
        mock_cpu_device.__str__.return_value = "CPU device 0"
        mock_devices.return_value = [mock_cpu_device]
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            print_gpu_info()
        
        output = captured_output.getvalue()
        assert "No GPU available - using CPU" in output
    
    @patch('jax.devices')
    def test_gpu_info_exception(self, mock_devices):
        """デバイス取得で例外が発生する場合のテスト"""
        mock_devices.side_effect = RuntimeError("JAX initialization failed")
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            print_gpu_info()
        
        output = captured_output.getvalue()
        assert "Cannot get GPU info:" in output
        assert "JAX initialization failed" in output
    
    @patch('jax.devices')
    def test_gpu_info_mixed_devices(self, mock_devices):
        """CPU・GPU混在時の情報表示テスト"""
        mock_cpu_device = MagicMock()
        mock_cpu_device.__str__.return_value = "CPU device 0"
        mock_gpu_device = MagicMock()
        mock_gpu_device.__str__.return_value = "GPU device 0"
        
        mock_devices.return_value = [mock_cpu_device, mock_gpu_device]
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            print_gpu_info()
        
        output = captured_output.getvalue()
        assert "Using GPU:" in output
        # 最初に見つかったGPUデバイスが表示される
        assert "GPU device 0" in output


class TestGPUUtilsIntegration:
    """GPU ユーティリティ統合テスト（実環境での動作確認用）"""
    
    @pytest.mark.skipif(
        not hasattr(sys, '_called_from_test'),
        reason="Integration test - run only in GPU environment"
    )
    def test_real_gpu_check(self):
        """実GPU環境での動作テスト（手動実行用）"""
        try:
            result = check_gpu_available()
            assert result is True
        except RuntimeError as e:
            pytest.skip(f"GPU not available: {e}")
    
    @pytest.mark.skipif(
        not hasattr(sys, '_called_from_test'),
        reason="Integration test - run only in GPU environment" 
    )
    def test_real_require_gpu_decorator(self):
        """実GPU環境でのデコレータテスト（手動実行用）"""
        @require_gpu()
        def gpu_test():
            import jax.numpy as jnp
            x = jnp.array([1.0, 2.0, 3.0])
            return float(jnp.sum(x))
        
        try:
            result = gpu_test()
            assert result == 6.0
        except SystemExit:
            pytest.skip("GPU not available for decorator test")


if __name__ == "__main__":
    # 統合テストを有効にするフラグ
    sys._called_from_test = True
    pytest.main([__file__, "-v"])