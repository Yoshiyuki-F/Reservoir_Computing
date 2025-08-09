#!/bin/bash
# Reservoir Computing GPU実行用ラッパー (uv対応)

# PATH設定
export PATH="$HOME/.local/bin:$PATH"

# GPU専用環境変数設定
unset LD_LIBRARY_PATH
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 引数の最初がpythonの場合、uv runに置換
if [ "$1" = "python" ]; then
    shift
    exec uv run python "$@"
else
    # pythonスクリプトファイルが直接渡された場合
    if [[ "$1" == *.py ]]; then
        exec uv run python "$@"
    else
        # その他のコマンドはそのまま実行
        exec "$@"
    fi
fi