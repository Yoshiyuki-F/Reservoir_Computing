# Modular Configuration System

This directory contains a modular configuration system that separates concerns into different categories, making it easy to mix and match components for different experiments.

## Directory Structure

```
presets/
├── datasets.json       # Dataset generation parameters
├── models/             # Model architecture configurations
│   ├── shared_reservoir_params.json
│   ├── quantum_advanced.json
│   ├── quantum_gate_based.json
│   ├── quantum_analog.json
│   ├── reservoir_complex.json
│   ├── reservoir_large.json
│   └── reservoir_standard.json
├── training/           # Training and preprocessing parameters
│   ├── robust.json
│   ├── standard.json
│   └── windowed.json
├── visualization/      # Visualization and plotting settings
│   ├── lorenz.json
│   ├── mackey_glass.json
│   └── sine_wave.json
└── experiments/        # Complete experiment compositions
    ├── lorenz_large.json
    ├── mackey_glass_windowed.json
    ├── sine_wave_quantum.json
    └── sine_wave_standard.json
```

## Benefits

1. **Separation of Concerns**: Dataset generation, model configuration, training parameters, and visualization settings are separate
2. **Reusability**: Same model configurations can be used with different datasets
3. **Composability**: Easy to create new experiment combinations
4. **Maintainability**: Changes to model configs affect all experiments using that model
5. **Clarity**: Each config file has a single, clear purpose

## Usage

### Programmatic Usage
```python
from models.reservoir.config import load_composed_config

# Load a complete experiment configuration
config = load_composed_config('sine_wave_standard')

# The config is automatically composed from:
# - datasets.json (sine_wave entry)
# - models/reservoir_standard.json
# - training/standard.json
# - visualization/sine_wave.json
```

### CLI Usage
```bash
# Use predefined experiment configurations
python __main__.py --dataset sine    # Uses sine_wave_standard
python __main__.py --dataset lorenz  # Uses lorenz_large
python __main__.py --dataset mackey  # Uses mackey_glass_windowed

# Quantum mode automatically switches to quantum configs when available
python __main__.py --dataset sine --quantum  # Uses sine_wave_quantum
```

## Creating New Configurations

1. **Add a new dataset**: Add a new entry to `datasets.json`
2. **Add a new model**: Create `models/my_model.json`
3. **Add a new training setup**: Create `training/my_training.json`
4. **Compose an experiment**: Create `experiments/my_experiment.json` referencing the above

Example new experiment:
```json
{
  "name": "my_experiment",
  "description": "My custom experiment",
  "dataset": "my_dataset",
  "model": "my_model",
  "training": "my_training",
  "visualization": "sine_wave"
}
```

## Migration from Legacy

The system automatically converts modular configs to the legacy format for backward compatibility with existing code.
