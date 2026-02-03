# Reservoir Computing with JAX

A JAX-based Reservoir Computing implementation designed for high-performance numerical computation and flexibility. This project supports various reservoir architectures, including classical Reservoirs and Quantum Reservoirs (Gate-based), and is capable of handling both Regression and Classification tasks.

## Features

- **JAX-based**: Optimized for high-speed numerical computation and GPU acceleration.
- **Unified Pipeline**: Consistent interface for running experiments across different models and datasets.
- **Multiple Architectures**:
    - Classical Reservoir
    - Feedforward Neural Network (FNN)
    - Distillation Models (FNN/RNN from Reservoir)
    - PassThrough Models (no train. just read out)
    - Quantum Reservoirs (Gate-based)
- **Flexible Data Handling**: Built-in support for synthetic datasets (Lorenz, Mackey-Glass) and MNIST.
- **GPU Support**: Built-in GPU detection and optimization.

## Installation

This project uses `uv` for dependency management.

```bash
# Clone the repository and navigate to the directory
cd /path/to/reservoir

# Install dependencies
uv sync --upgrade

# Activate virtual environment
source .venv/bin/activate
```

## Usage

The project provides a unified CLI entry point `reservoir-cli`.

### Basic Command

```bash
uv run reservoir-cli --model <MODEL> --dataset <DATASET> [OPTIONS]
```

### Arguments

- `--model`: **(Required)** The model architecture to use.
    - `classical_reservoir`
    - `fnn`
    - `fnn_distillation`
    - `quantum_reservoir`
    - `passthrough`
- `--dataset`: **(Required)** The dataset to use.
    - `lorenz` (Regression)
    - `mackey_glass` (Regression)
    - `mnist` (Classification)
- `--force-cpu`: Force execution on CPU (not tried)

### Examples

**1. Classical Reservoir on Sine Wave**
```bash
uv run reservoir-cli --model classical_reservoir --dataset mnist
```

**2. Quantum Gate-based Reservoir on Lorenz Attractor**
```bash
uv run reservoir-cli --model quantum_reservoir --dataset lorenz
```

**3. MLP Training (FNN) on MNIST**
```bash
uv run reservoir-cli --model fnn --dataset mnist
```

### GPU & Testing with Poe

GPU execution is recommended for performance.

```bash
# Run tests (CPU)
uv run poe test

# Run GPU smoke tests
uv run poe test-gpu

# Run CLI with GPU environment variables set
uv run poe cli-gpu -- --model classical_reservoir --dataset sine_wave
```

## File Structure

```
.
├── src/reservoir/
│   ├── cli/                # CLI entry point (main.py)
│   ├── core/               # Core definitions (identifiers, config)
│   ├── models/             # Model architectures and presets
│   ├── pipelines/          # Experiment pipelines
│   ├── layers/             # Neural network layers (Reservoir, Attention, etc.)
│   ├── data/               # Data generation and loading
│   └── utils/              # Utilities (GPU, metrics, logging)
├── tests/                  # Unit and integration tests
├── pyproject.toml          # Project configuration and dependencies
└── README.md
```

## References

- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
- [JAX Documentation](https://jax.readthedocs.io/)
- [TensorCircuit](https://github.com/tensorcircuit/tensorcircuit-ng/) (for Quantum Reservoir Computing)
