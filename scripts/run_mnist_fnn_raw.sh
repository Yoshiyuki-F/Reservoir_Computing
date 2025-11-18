#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <hidden_dim> [additional reservoir-cli args]" >&2
    echo "  example: $0 100   # 100-unit hidden layer (raw, no FS/DMB)" >&2
    exit 1
fi

hidden_dim="$1"
shift || true

if ! [[ "$hidden_dim" =~ ^[0-9]+$ ]]; then
    echo "hidden_dim must be an integer, got '$hidden_dim'" >&2
    exit 1
fi

config_path="outputs/mnist_fnn_raw_${hidden_dim}.json"
mkdir -p "$(dirname "$config_path")"

cat > "$config_path" <<EOF
{
  "model": {
    "layer_dims": [784, ${hidden_dim}, 10]
  },
  "training": {
    "learning_rate": 0.001,
    "batch_size": 128,
    "num_epochs": 20,
    "weights_path": "outputs/mnist_fnn_raw_${hidden_dim}.msgpack"
  },
  "ridge_lambdas": [-7, 7, 15],
  "use_preprocessing": false
}
EOF

unset LD_LIBRARY_PATH || true
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

uv run reservoir-cli \
    --dataset mnist \
    --model fnn_pretrained \
    --config "$config_path" \
    "$@"
