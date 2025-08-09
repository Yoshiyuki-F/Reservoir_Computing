#!/usr/bin/env python3
"""
Reservoir Computing実装の基本テスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reservoir import ReservoirComputer
from reservoir.utils import generate_sine_data, calculate_mse

def test_basic_functionality():
    """基本機能のテスト"""
    print("=== Reservoir Computing基本機能テスト ===")
    
    # 小さなデータセットでテスト
    input_data, target_data = generate_sine_data(time_steps=200)
    print(f"データ形状: input={input_data.shape}, target={target_data.shape}")
    
    # 小さなreservoirでテスト
    rc = ReservoirComputer(
        n_inputs=1, 
        n_reservoir=50, 
        n_outputs=1,
        spectral_radius=0.9,
        random_seed=42
    )
    
    print("Reservoir情報:", rc.get_reservoir_info())
    
    # 訓練
    print("訓練中...")
    rc.train(input_data, target_data)
    
    # 予測
    print("予測中...")
    predictions = rc.predict(input_data[:50])
    
    # 誤差計算
    mse = calculate_mse(predictions, target_data[:50])
    print(f"予測形状: {predictions.shape}")
    print(f"MSE: {mse:.6f}")
    
    print("基本機能テスト完了!")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        print("\n すべてのテストが成功しました！")
    except Exception as e:
        print(f" エラーが発生しました: {e}")
        import traceback
        traceback.print_exc() 