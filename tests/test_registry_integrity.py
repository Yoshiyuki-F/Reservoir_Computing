"""
Sanity Check Suite for Registry Integrity
Phase 4 - Process 3 (SSOT Verification)

Objective:
Ensure all registered presets (Datasets, Models, Training) adhere to V2 Strict Standards.
This prevents the pipeline from crashing at runtime due to missing metadata (e.g., n_input).
"""

import pytest
import math

# Imports from the Project
from reservoir.data.presets import DATASET_REGISTRY, DatasetPreset
# Assuming these exist based on Phase 4 architecture
from reservoir.models.presets import get_model_preset, MODEL_PRESETS
from reservoir.training.presets import get_training_preset, TRAINING_PRESETS


class TestDatasetRegistryIntegrity:
    """Verifies src/reservoir/data/presets.py"""

    def test_all_datasets_have_io_dims(self):
        """
        CRITICAL: All datasets MUST define n_input and n_output.
        run.py relies on this to build the model topology.
        """
        for name in DATASET_REGISTRY.list_keys():
            preset = DATASET_REGISTRY.get(name)
            assert isinstance(preset, DatasetPreset), f"{name} is not a DatasetPreset instance"

            # Check Input Dimension
            assert preset.config.n_input is not None, \
                f"Dataset '{name}' is missing 'n_input'. Strict Mode requires explicit dimensions."
            assert isinstance(preset.config.n_input, int) and preset.config.n_input > 0, \
                f"Dataset '{name}' 'n_input' must be a positive integer."

            # Check Output Dimension
            assert preset.config.n_output is not None, \
                f"Dataset '{name}' is missing 'n_output'. Strict Mode requires explicit dimensions."
            assert isinstance(preset.config.n_output, int) and preset.config.n_output > 0, \
                f"Dataset '{name}' 'n_output' must be a positive integer."

    def test_dataset_aliases_resolve(self):
        """Ensure all aliases point to valid canonical names."""
        # Access internal aliases dict if exposed, or test known aliases
        # Assuming generic verify via registry access
        known_aliases = ["sine", "sw", "m"]  # from your provided snippet
        for alias in known_aliases:
            preset = DATASET_REGISTRY.get(alias)
            assert preset is not None, f"Alias '{alias}' failed to resolve."


class TestModelRegistryIntegrity:
    """Verifies src/reservoir/models/presets.py"""

    def test_model_physics_completeness(self):
        """
        Reservoir presets must define core physics parameters.
        """
        required_params = ["n_units", "leak_rate", "spectral_radius", "input_scale"]

        # Iterate over all defined model presets
        # If MODEL_PRESETS is a dict of definitions
        keys = MODEL_PRESETS.keys() if isinstance(MODEL_PRESETS, dict) else []

        for name in keys:
            preset = get_model_preset(name)
            params = preset.to_params()

            for param in required_params:
                assert param in params, f"Model '{name}' is missing core parameter: {param}"
                assert params[param] is not None, f"Model '{name}' parameter {param} cannot be None"

    def test_forbidden_parameters(self):
        """
        BANNED: 'alpha' is ambiguous. Use 'leak_rate' (reservoir) or 'ridge_lambda' (readout).
        """
        keys = MODEL_PRESETS.keys() if isinstance(MODEL_PRESETS, dict) else []
        for name in keys:
            preset = get_model_preset(name)
            params = preset.to_params()
            assert "alpha" not in params, \
                f"Model '{name}' uses forbidden parameter 'alpha'. Refactor to 'leak_rate'."

    def test_parameter_sanity(self):
        """Check for mathematical validity (e.g. leak_rate in (0, 1])"""
        keys = MODEL_PRESETS.keys() if isinstance(MODEL_PRESETS, dict) else []
        for name in keys:
            preset = get_model_preset(name)
            lr = preset.leak_rate
            if lr is not None:
                assert 0.0 < lr <= 1.0, f"Model '{name}' has invalid leak_rate: {lr}"


class TestTrainingRegistryIntegrity:
    """Verifies src/reservoir/training/presets.py"""

    def test_ridge_lambda_naming(self):
        """
        Ensure we use 'ridge_lambda' not 'alpha' or 'reg'.
        """
        keys = TRAINING_PRESETS.keys() if isinstance(TRAINING_PRESETS, dict) else []
        for name in keys:
            preset = get_training_preset(name)
            # Assuming preset converts to dict or has attributes
            cfg = preset.to_dict() if hasattr(preset, "to_dict") else preset.__dict__

            assert "ridge_lambda" in cfg, f"Training preset '{name}' missing 'ridge_lambda'."
            assert "alpha" not in cfg, f"Training preset '{name}' contains forbidden 'alpha'."