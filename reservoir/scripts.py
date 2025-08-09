#!/usr/bin/env python3
"""
Reservoir Computing GPU実行用スクリプト
"""
import os
import sys
import subprocess


def setup_gpu_environment():
    """GPU専用環境変数を設定"""
    # LD_LIBRARY_PATH競合を回避
    if 'LD_LIBRARY_PATH' in os.environ:
        del os.environ['LD_LIBRARY_PATH']
    
    # JAX GPU専用設定
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def run_with_gpu():
    """汎用GPU実行ラッパー"""
    setup_gpu_environment()
    
    if len(sys.argv) < 2:
        print("使用方法: reservoir-gpu <python_script> [args...]")
        print("例: reservoir-gpu examples/demo.py")
        sys.exit(1)
    
    # プロジェクトルートディレクトリに移動
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # 引数をそのまま python に渡す
    cmd = [sys.executable] + sys.argv[1:]
    subprocess.run(cmd)


def run_demo_gpu():
    """デモをGPUで実行"""
    setup_gpu_environment()
    
    # プロジェクトルートディレクトリに移動
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # examples/demo.py を実行
    cmd = [sys.executable, "examples/demo.py"]
    subprocess.run(cmd)


def run_tests_gpu():
    """テストをGPUで実行"""
    setup_gpu_environment()
    
    # プロジェクトルートディレクトリに移動
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # 基本GPU動作確認
    cmd = [sys.executable, "tests/test_cuda.py"]
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n GPU動作確認成功")
        
        # Reservoir Computing テスト
        cmd = [sys.executable, "tests/test_simple.py"]
        subprocess.run(cmd)
    else:
        print(" GPU動作確認失敗")
        sys.exit(1)


if __name__ == "__main__":
    run_with_gpu()