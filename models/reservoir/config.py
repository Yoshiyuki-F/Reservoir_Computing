"""
Reservoir-specific configuration classes.
"""

from configs.core.config import ModelConfig


class ReservoirConfig(ModelConfig):
    """Reservoir Computer hyperparameter configuration."""

    def __init__(self, **data):
        # Convert reservoir-specific fields to the generic model format
        if 'model_type' not in data:
            data['model_type'] = 'reservoir'
        if 'name' not in data and 'model_type' in data:
            data['name'] = f"{data['model_type']}_config"

        # Move reservoir-specific fields to params
        reservoir_fields = {
            'n_inputs', 'n_reservoir', 'n_outputs', 'spectral_radius',
            'input_scaling', 'noise_level', 'alpha', 'random_seed',
            'reservoir_weight_range', 'sparsity', 'input_bias', 'nonlinearity'
        }

        params = data.get('params', {})
        for field in list(data.keys()):
            if field in reservoir_fields:
                params[field] = data.pop(field)
        data['params'] = params

        super().__init__(**data)


class QuantumReservoirConfig(ModelConfig):
    """Quantum Reservoir Computer configuration."""

    def __init__(self, **data):
        # Convert quantum-specific fields to the generic model format
        if 'model_type' not in data:
            data['model_type'] = 'quantum'
        if 'name' not in data and 'model_type' in data:
            data['name'] = f"{data['model_type']}_config"

        # Move quantum-specific fields to params
        quantum_fields = {
            'n_qubits', 'circuit_depth', 'n_inputs', 'n_outputs',
            'entanglement_pattern', 'measurement_basis', 'random_seed'
        }

        params = data.get('params', {})
        for field in list(data.keys()):
            if field in quantum_fields:
                params[field] = data.pop(field)
        data['params'] = params

        super().__init__(**data)