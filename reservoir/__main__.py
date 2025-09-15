"""
Reservoir computing module entry point.
Allows running: python -m reservoir.cli
"""

import argparse
import os
import sys
from pathlib import Path

from reservoir.runner import run_experiment_from_config


def main():
    """メイン関数。"""
    parser = argparse.ArgumentParser(
        description="JAXを使ったReservoir Computingのデモンストレーション",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        使用例:
          # 設定ファイルを指定してデモを実行
          python -m reservoir.__main__ --config configs/sine_wave_demo_config.json

          # 複数の設定ファイルを一度に実行
          python -m reservoir.__main__ --config configs/sine_wave_demo_config.json configs/lorenz_demo_config.json

          # すべてのデフォルトデモを実行（デフォルト動作）
          python -m reservoir.__main__
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        nargs='*',
        help='設定ファイルのパス (例: configs/sine_wave_demo_config.json)。複数指定可能'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='すべてのデフォルトデモを実行'
    )

    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='CPUを強制使用（デフォルトではGPUが必須）'
    )

    args = parser.parse_args()

    print("JAXを使ったReservoir Computingのデモンストレーション")
    print("=" * 60)

    # GPU環境の事前確認
    if not args.force_cpu:
        try:
            from utils.gpu_utils import check_gpu_available
            check_gpu_available()  # GPU必須チェック（詳細情報も表示）
        except Exception:
            sys.exit(1)  # gpu_utils が既に詳細なエラー情報を表示済み
    else:
        print("\nCPU実行モードが指定されました")

    try:
        backend = 'cpu' if args.force_cpu else 'gpu'

        if args.config:
            # 設定ファイルが指定された場合
            config_files_to_run = args.config
        else:
            # デフォルト動作またはallフラグ: すべてのデモを実行
            if args.all:
                print("すべてのデフォルトデモを実行します...")
            else:
                print("設定が指定されていません。すべてのデフォルトデモを実行します...")
                print("(特定の設定ファイルを使用するには--configオプションを使用してください)")

            config_files_to_run = [
                'configs/sine_wave_demo_config.json',
                'configs/lorenz_demo_config.json',
                'configs/mackey_glass_demo_config.json'
            ]

        results = []
        for config_file in config_files_to_run:
            # 絶対パスに変換
            abs_config_path = Path(__file__).parent.parent / config_file
            if not abs_config_path.exists():
                print(f"エラー: 設定ファイルが見つかりません: {abs_config_path}")
                sys.exit(1)

            result = run_experiment_from_config(abs_config_path, backend)
            config_name = os.path.basename(config_file)
            results.append((f"実験 ({config_name})", result))

        # 結果サマリーを表示
        if len(results) > 0:
            print("\n" + "=" * 60)
            print("デモンストレーションが完了しました")
            print("=" * 60)
            print("結果サマリー:")

            for demo_type, (train_mse, test_mse, train_mae, test_mae) in results:
                if train_mse is not None:  # 訓練結果がある場合
                    print(f"{demo_type} - 訓練 MSE: {train_mse:.6f}, テスト MSE: {test_mse:.6f}")
                else:
                    print(f"{demo_type} - テスト MSE: {test_mse:.6f}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    main()