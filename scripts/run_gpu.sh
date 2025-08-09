#!/bin/bash
# Reservoir Computing GPU実行用ラッパー

# GPU専用環境変数設定
unset LD_LIBRARY_PATH
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 引数をそのまま渡して実行
exec "$@"