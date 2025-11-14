#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <n_reservoir> [additional reservoir-cli args]" >&2
    exit 1
fi

n_reservoir="$1"
shift || true

if ! [[ "$n_reservoir" =~ ^[0-9]+$ ]]; then
    echo "n_reservoir must be an integer, got '$n_reservoir'" >&2
    exit 1
fi

unset LD_LIBRARY_PATH || true
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

uv run reservoir-cli \
    --dataset mnist \
    --model classic_standard \
    --training raw_standard \
    --n-reservoir "$n_reservoir" \
    "$@"
