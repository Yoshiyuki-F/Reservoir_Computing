"""
Reservoir-specific configuration classes and validation utilities.
"""

from typing import Any, Dict, Sequence, Set, Union, Tuple, ClassVar
import numpy as np

from core_lib.core.config import ModelConfig


# ============================================================================
# Validation utilities
# ============================================================================

def parse_ridge_lambdas(
    cfg: Dict[str, Any],
    key: str = "ridge_lambdas",
    default: Union[Tuple[float, ...], list, None] = None
) -> Tuple[float, ...]:
    """Parse and validate ridge_lambdas from configuration.

    Supports three formats:
    1. [start, stop, num] or (start, stop, num) -> np.logspace(start, stop, num)
    2. {"start": -14, "stop": 2, "num": 25} -> dict format
    3. [1e-6, 1e-4, 1e-2, ...] -> explicit list of values

    Args:
        cfg: Configuration dictionary
        key: Key to look up in cfg (default: "ridge_lambdas")
        default: Default value if key not found. If None and key missing, raises error.
                Can be a tuple/list of explicit values or [start, stop, num] format.

    Returns:
        Tuple of lambda values

    Raises:
        ValueError: If ridge_lambdas is missing (and no default), invalid, or empty
    """
    if key not in cfg:
        if default is None:
            raise ValueError(
                f"{key} is required in configuration. "
                "Specify as [start, stop, num] for logspace (e.g., [-14, 2, 25]) "
                "or as explicit list of values (e.g., [1e-6, 1e-4, 1e-2])"
            )
        # Use default value - recursively parse it
        return parse_ridge_lambdas({key: default}, key=key, default=None)

    ridge_cfg = cfg[key]
    if not isinstance(ridge_cfg, (list, tuple, dict)):
        raise ValueError(
            f"{key} must be a list, tuple, or dict. "
            f"Got: {type(ridge_cfg).__name__}"
        )

    if isinstance(ridge_cfg, (list, tuple)) and len(ridge_cfg) == 3:
        # Format: [start, stop, num] or (start, stop, num)
        try:
            start, stop, num = ridge_cfg
            start, stop, num = float(start), float(stop), int(num)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{key} format [start, stop, num] requires numeric values. "
                f"Got: {ridge_cfg}. Error: {e}"
            )
        if num <= 0:
            raise ValueError(f"{key} num must be positive, got: {num}")
        return tuple(float(val) for val in np.logspace(start, stop, num))

    elif isinstance(ridge_cfg, dict):
        # Format: {"start": -14, "stop": 2, "num": 25}
        if not all(k in ridge_cfg for k in ["start", "stop", "num"]):
            raise ValueError(
                f"{key} dict format requires 'start', 'stop', and 'num' keys"
            )
        try:
            start = float(ridge_cfg["start"])
            stop = float(ridge_cfg["stop"])
            num = int(ridge_cfg["num"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{key} dict values must be numeric. Error: {e}"
            )
        if num <= 0:
            raise ValueError(f"{key} num must be positive, got: {num}")
        return tuple(float(val) for val in np.logspace(start, stop, num))

    else:
        # Explicit list of lambda values
        if not ridge_cfg:
            raise ValueError(f"{key} cannot be empty")
        try:
            return tuple(float(l) for l in ridge_cfg)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{key} must contain numeric values. "
                f"Got: {ridge_cfg}. Error: {e}"
            )


def validate_positive_int(cfg: Dict[str, Any], key: str, name: str = None) -> int:
    """Validate and extract a positive integer from configuration.

    Args:
        cfg: Configuration dictionary
        key: Key to look up in cfg
        name: Human-readable name for error messages (defaults to key)

    Returns:
        Validated positive integer

    Raises:
        ValueError: If value is missing, not an integer, or not positive
    """
    name = name or key
    if key not in cfg:
        raise ValueError(f"{name} is required in configuration")

    try:
        value = int(cfg[key])
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an integer. Got: {cfg[key]}. Error: {e}")

    if value < 1:
        raise ValueError(f"{name} must be at least 1, got: {value}")

    return value


def validate_positive_float(cfg: Dict[str, Any], key: str, name: str = None) -> float:
    """Validate and extract a positive float from configuration.

    Args:
        cfg: Configuration dictionary
        key: Key to look up in cfg
        name: Human-readable name for error messages (defaults to key)

    Returns:
        Validated positive float

    Raises:
        ValueError: If value is missing, not numeric, or not positive
    """
    name = name or key
    if key not in cfg:
        raise ValueError(f"{name} is required in configuration")

    try:
        value = float(cfg[key])
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be numeric. Got: {cfg[key]}. Error: {e}")

    if value <= 0:
        raise ValueError(f"{name} must be positive, got: {value}")

    return value


def validate_enum(cfg: Dict[str, Any], key: str, allowed: Set[str], name: str = None) -> str:
    """Validate and extract an enum value from configuration.

    Args:
        cfg: Configuration dictionary
        key: Key to look up in cfg
        allowed: Set of allowed values
        name: Human-readable name for error messages (defaults to key)

    Returns:
        Validated enum value (lowercased)

    Raises:
        ValueError: If value is missing or not in allowed set
    """
    name = name or key
    if key not in cfg:
        raise ValueError(f"{name} is required in configuration")

    value = str(cfg[key]).lower()
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {sorted(allowed)}, got: '{value}'"
        )

    return value


def validate_sequence(cfg: Dict[str, Any], key: str, name: str = None) -> Sequence[Any]:
    """Validate and extract a sequence from configuration.

    Args:
        cfg: Configuration dictionary
        key: Key to look up in cfg
        name: Human-readable name for error messages (defaults to key)

    Returns:
        Validated sequence

    Raises:
        ValueError: If value is missing or not a sequence
    """
    name = name or key
    if key not in cfg:
        raise ValueError(f"{name} is required in configuration")

    value = cfg[key]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple. Got: {type(value).__name__}")

    if not value:
        raise ValueError(f"{name} cannot be empty")

    return value


class ReservoirConfig(ModelConfig):
    """Reservoir Computer hyperparameter configuration with full validation."""

    SUPPORTED_STATE_AGG: ClassVar[Set[str]] = {"last", "mean", "concat", "last_mean", "mts"}

    def __init__(self, **data):
        # Convert reservoir-specific fields to the generic model format
        if 'model_type' not in data:
            data['model_type'] = 'reservoir'
        if 'name' not in data and 'model_type' in data:
            data['name'] = f"{data['model_type']}_config"

        # Move reservoir-specific fields to params
        reservoir_fields = {
            'n_inputs', 'n_hiddenLayer', 'n_outputs', 'spectral_radius',
            'input_scaling', 'noise_level', 'alpha', 'random_seed',
            'reservoir_weight_range', 'sparsity', 'input_bias', 'nonlinearity',
            'state_aggregation', 'ridge_lambdas', 'use_preprocessing',
            'include_bias', 'poly_mode', 'poly_degree', 'std_threshold',
            'readout_cv', 'readout_n_folds', 'washout_steps'
        }

        params = data.get('params', {})
        for field in list(data.keys()):
            if field in reservoir_fields:
                params[field] = data.pop(field)
        data['params'] = params

        # ===== VALIDATION STARTS HERE =====

        # Validate required keys
        required_keys = {
            'n_inputs', 'n_hiddenLayer', 'n_outputs', 'spectral_radius',
            'input_scaling', 'noise_level', 'alpha', 'reservoir_weight_range',
            'sparsity', 'input_bias', 'nonlinearity', 'random_seed', 'state_aggregation',
            'ridge_lambdas'
        }
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
            formatted_keys = ", ".join(sorted(missing_keys))
            raise ValueError(f"Classical reservoir configuration missing required keys: {formatted_keys}")

        # Validate positive integers
        n_inputs = params['n_inputs']
        if not isinstance(n_inputs, int) or n_inputs <= 0:
            raise ValueError(f"n_inputs must be a positive integer, got: {n_inputs}")

        n_hiddenLayer = params['n_hiddenLayer']
        if not isinstance(n_hiddenLayer, int) or n_hiddenLayer <= 0:
            raise ValueError(f"n_hiddenLayer must be a positive integer, got: {n_hiddenLayer}")

        n_outputs = params['n_outputs']
        if not isinstance(n_outputs, int) or n_outputs <= 0:
            raise ValueError(f"n_outputs must be a positive integer, got: {n_outputs}")

        # Validate state_aggregation
        state_aggregation = str(params.get('state_aggregation', 'last')).lower()
        if state_aggregation not in self.SUPPORTED_STATE_AGG:
            raise ValueError(f"state_aggregation must be one of {sorted(self.SUPPORTED_STATE_AGG)}, got: '{state_aggregation}'")

        super().__init__(**data)


class QuantumReservoirConfig(ModelConfig):
    """Quantum Reservoir Computer configuration with full validation."""

    SUPPORTED_MEASUREMENT_BASIS: ClassVar[Set[str]] = {"pauli-z", "multi-pauli"}
    SUPPORTED_ENCODING: ClassVar[Set[str]] = {"amplitude", "angle", "detuning"}
    SUPPORTED_ENTANGLEMENT: ClassVar[Set[str]] = {"circular", "full"}
    SUPPORTED_STATE_AGG: ClassVar[Set[str]] = {"last", "mean", "concat", "last_mean", "mts"}

    def __init__(self, **data):
        # Convert quantum-specific fields to the generic model format
        if 'model_type' not in data:
            data['model_type'] = 'quantum'
        if 'name' not in data and 'model_type' in data:
            data['name'] = f"{data['model_type']}_config"

        # Move quantum-specific fields to params
        quantum_fields = {
            'n_qubits', 'circuit_depth', 'n_inputs', 'n_outputs',
            'entanglement_pattern', 'measurement_basis', 'random_seed',
            'backend', 'encoding_scheme', 'entanglement', 'detuning_scale',
            'state_aggregation', 'readout_observables', 'ridge_lambdas'
        }

        params = data.get('params', {})
        for field in list(data.keys()):
            if field in quantum_fields:
                params[field] = data.pop(field)
        data['params'] = params

        # ===== VALIDATION STARTS HERE =====

        # Validate required keys
        required_keys = {
            'n_qubits', 'circuit_depth', 'n_inputs', 'n_outputs',
            'backend', 'random_seed', 'measurement_basis', 'encoding_scheme',
            'entanglement', 'detuning_scale', 'state_aggregation', 'ridge_lambdas'
        }
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
            formatted_keys = ", ".join(sorted(missing_keys))
            raise ValueError(f"Gate-based quantum reservoir configuration missing required keys: {formatted_keys}")

        # Validate positive integers
        n_qubits = params['n_qubits']
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError(f"n_qubits must be a positive integer, got: {n_qubits}")

        circuit_depth = params['circuit_depth']
        if not isinstance(circuit_depth, int) or circuit_depth <= 0:
            raise ValueError(f"circuit_depth must be a positive integer, got: {circuit_depth}")

        n_inputs = params['n_inputs']
        if not isinstance(n_inputs, int) or n_inputs <= 0:
            raise ValueError(f"n_inputs must be a positive integer, got: {n_inputs}")

        n_outputs = params['n_outputs']
        if not isinstance(n_outputs, int) or n_outputs <= 0:
            raise ValueError(f"n_outputs must be a positive integer, got: {n_outputs}")

        # Validate measurement_basis
        measurement_basis = str(params['measurement_basis']).lower()
        if measurement_basis not in self.SUPPORTED_MEASUREMENT_BASIS:
            raise ValueError(f"measurement_basis must be one of {sorted(self.SUPPORTED_MEASUREMENT_BASIS)}, got: '{measurement_basis}'")

        # Validate encoding_scheme
        encoding_scheme = str(params['encoding_scheme']).lower()
        if encoding_scheme not in self.SUPPORTED_ENCODING:
            raise ValueError(f"encoding_scheme must be one of {sorted(self.SUPPORTED_ENCODING)}, got: '{encoding_scheme}'")

        # Validate entanglement
        entanglement = str(params['entanglement']).lower()
        if entanglement == "ring":
            entanglement = "circular"
            params['entanglement'] = "circular"
        if entanglement not in self.SUPPORTED_ENTANGLEMENT:
            raise ValueError(f"entanglement must be one of {sorted(self.SUPPORTED_ENTANGLEMENT)}, got: '{entanglement}'")

        # Validate state_aggregation
        state_aggregation = str(params['state_aggregation']).lower()
        if state_aggregation not in self.SUPPORTED_STATE_AGG:
            raise ValueError(f"state_aggregation must be one of {sorted(self.SUPPORTED_STATE_AGG)}, got: '{state_aggregation}'")

        super().__init__(**data)


class AnalogQuantumReservoirConfig(ModelConfig):
    """Analog quantum reservoir configuration with full validation."""

    SUPPORTED_ENCODING: ClassVar[Set[str]] = {"detuning"}
    SUPPORTED_MEASUREMENTS: ClassVar[Set[str]] = {"multi-pauli"}
    SUPPORTED_INPUT_MODES: ClassVar[Set[str]] = {"scalar", "sequence", "block"}
    SUPPORTED_STATE_AGG: ClassVar[Set[str]] = {"last", "mean", "last_mean", "mts"}
    ALLOWED_READOUT_OBSERVABLES: ClassVar[Set[str]] = {"X", "Y", "Z", "ZZ"}

    def __init__(self, **data):
        if 'model_type' not in data:
            data['model_type'] = 'analog_quantum'
        if 'name' not in data and 'model_type' in data:
            data['name'] = f"{data['model_type']}_config"

        analog_fields = {
            'n_qubits',
            'positions',
            'C6',
            'Omega',
            'Delta_g',
            'Delta_l',
            't_final',
            'dt',
            'encoding_scheme',
            'measurement_basis',
            'readout_observables',
            'state_aggregation',
            'reupload_layers',
            'input_mode',
            'detuning_scale',
            'random_seed',
            'ridge_lambdas',
        }

        params = data.get('params', {})
        for field in list(data.keys()):
            if field in analog_fields:
                params[field] = data.pop(field)
        data['params'] = params

        # ===== VALIDATION STARTS HERE =====

        # Validate model_type
        model_type = data.get('model_type', 'analog_quantum')
        if model_type != 'analog_quantum':
            raise ValueError(f"AnalogQuantumReservoirConfig expects model_type='analog_quantum', got '{model_type}'")

        # Validate required keys
        required_keys = {
            'n_qubits', 'C6', 'Omega', 'Delta_g', 'Delta_l',
            't_final', 'dt', 'encoding_scheme', 'measurement_basis',
            'readout_observables', 'state_aggregation', 'reupload_layers',
            'input_mode', 'detuning_scale', 'ridge_lambdas'
        }
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
            formatted_keys = ", ".join(sorted(missing_keys))
            raise ValueError(f"AnalogQuantumReservoir missing required configuration keys: {formatted_keys}")

        # Validate n_qubits (positive integer)
        n_qubits = params['n_qubits']
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError(f"n_qubits must be a positive integer, got: {n_qubits}")

        # Validate positions shape if provided
        if 'positions' in params and params['positions'] is not None:
            positions = np.asarray(params['positions'], dtype=np.float64)
            if positions.shape[0] != n_qubits:
                raise ValueError(f"positions must have shape ({n_qubits}, ...), got shape {positions.shape}")

            # Validate positions are distinct
            for j in range(n_qubits):
                for k in range(j + 1, n_qubits):
                    diff = positions[j] - positions[k]
                    dist = np.linalg.norm(diff)
                    if dist <= 0.0:
                        raise ValueError("Qubit positions must be distinct")

        # Validate t_final and dt (positive)
        t_final = float(params['t_final'])
        dt = float(params['dt'])
        if t_final <= 0 or dt <= 0:
            raise ValueError("t_final and dt must be positive")

        # Validate encoding_scheme
        encoding_scheme = str(params['encoding_scheme']).lower()
        if encoding_scheme not in self.SUPPORTED_ENCODING:
            raise ValueError(f"encoding_scheme '{encoding_scheme}' not supported. Supported: {sorted(self.SUPPORTED_ENCODING)}")

        # Validate measurement_basis
        measurement_basis = str(params['measurement_basis']).lower()
        if measurement_basis not in self.SUPPORTED_MEASUREMENTS:
            raise ValueError(f"measurement_basis '{measurement_basis}' not supported. Supported: {sorted(self.SUPPORTED_MEASUREMENTS)}")

        # Validate readout_observables
        raw_readouts = params['readout_observables']
        if not isinstance(raw_readouts, (list, tuple)):
            raise ValueError("readout_observables must be a sequence of observables")
        if not raw_readouts:
            raise ValueError("readout_observables must contain at least one entry")

        for entry in raw_readouts:
            name = str(entry).upper()
            if name not in self.ALLOWED_READOUT_OBSERVABLES:
                raise ValueError(f"Unsupported readout observable '{entry}'. Supported values: {sorted(self.ALLOWED_READOUT_OBSERVABLES)}")

        # Validate state_aggregation
        state_aggregation = str(params['state_aggregation']).lower()
        if state_aggregation not in self.SUPPORTED_STATE_AGG:
            raise ValueError(f"state_aggregation must be one of {sorted(self.SUPPORTED_STATE_AGG)}, got: '{state_aggregation}'")

        # Validate reupload_layers (positive integer)
        reupload_layers = params['reupload_layers']
        if not isinstance(reupload_layers, int) or reupload_layers <= 0:
            raise ValueError(f"reupload_layers must be a positive integer, got: {reupload_layers}")

        # Validate input_mode
        input_mode = str(params['input_mode']).lower()
        if input_mode not in self.SUPPORTED_INPUT_MODES:
            raise ValueError(f"input_mode must be one of {sorted(self.SUPPORTED_INPUT_MODES)}, got: '{input_mode}'")

        super().__init__(**data)


# ============================================================================
# Centralized error messages
# ============================================================================

class ErrorMessages:
    """Centralized error message templates for reservoir models."""

    # Configuration errors
    MISSING_REQUIRED_KEYS = "{model_name} missing required configuration keys: {keys}"
    WRONG_MODEL_TYPE = "{model_name} expects model_type='{expected_type}'"

    # Training/prediction errors
    NOT_TRAINED = "Model has not been trained. Call train() first."
    NOT_TRAINED_JP = "モデルが訓練されていません。先にtrain()を呼び出してください。"
    NOT_ENOUGH_SAMPLES = "Not enough samples to perform ridge regression"
    FEATURE_SCALER_NOT_FITTED = "Feature scaler has not been fitted. Call train() before predict()."
    NORMALIZATION_NOT_FITTED = "Normalization statistics are not fitted"
    DATA_LENGTH_MISMATCH = "input_data and target_data length mismatch"
    SEQUENCE_LABEL_MISMATCH = "Sequence/label length mismatch"

    # Classification errors
    CLASSIFICATION_NOT_ENABLED = "Classification mode not enabled. Call train_classification first."
    CLASSIFICATION_MODEL_NOT_TRAINED = "Classification model not trained"

    # Shape/dimension errors
    SHAPE_MISMATCH = "{param} must have shape {expected_shape}"
    INPUT_EXCEEDS_CAPACITY = "Input dimension {n_input} exceeds capacity {capacity}"

    # Validation errors (sequence/list)
    MUST_BE_SEQUENCE = "{param} must be a sequence of {content}"
    MUST_NOT_BE_EMPTY = "{param} must contain at least one entry"
    EMPTY_SEQUENCE = "{param} cannot be empty"

    # Enum/choice validation errors
    UNSUPPORTED_VALUE = "Unsupported {param} '{value}'. Supported values: {allowed}"
    MUST_BE_ONE_OF = "{param} must be one of {allowed}, got: '{value}'"

    # Numeric validation errors
    MUST_BE_POSITIVE = "{param} must be positive, got: {value}"
    MUST_BE_DISTINCT = "{param} must be distinct"

    # State/runtime errors
    STATE_NOT_SET = "{param} must be set before {operation}"

    @staticmethod
    def format_missing_keys(model_name: str, keys: Sequence[str]) -> str:
        """Format missing keys error message."""
        formatted_keys = ", ".join(sorted(keys))
        return ErrorMessages.MISSING_REQUIRED_KEYS.format(
            model_name=model_name,
            keys=formatted_keys
        )

    @staticmethod
    def format_wrong_model_type(model_name: str, expected_type: str) -> str:
        """Format wrong model type error message."""
        return ErrorMessages.WRONG_MODEL_TYPE.format(
            model_name=model_name,
            expected_type=expected_type
        )

    @staticmethod
    def format_shape_mismatch(param: str, expected_shape: str) -> str:
        """Format shape mismatch error message."""
        return ErrorMessages.SHAPE_MISMATCH.format(
            param=param,
            expected_shape=expected_shape
        )

    @staticmethod
    def format_input_exceeds_capacity(n_input: int, n_qubits: int, reupload_layers: int) -> str:
        """Format input exceeds capacity error message."""
        capacity = f"{n_qubits}×{reupload_layers}"
        return ErrorMessages.INPUT_EXCEEDS_CAPACITY.format(
            n_input=n_input,
            capacity=capacity
        )

    @staticmethod
    def format_must_be_sequence(param: str, content: str) -> str:
        """Format must be sequence error message."""
        return ErrorMessages.MUST_BE_SEQUENCE.format(
            param=param,
            content=content
        )

    @staticmethod
    def format_must_not_be_empty(param: str) -> str:
        """Format must not be empty error message."""
        return ErrorMessages.MUST_NOT_BE_EMPTY.format(param=param)

    @staticmethod
    def format_unsupported_value(param: str, value: str, allowed: Set[str]) -> str:
        """Format unsupported value error message."""
        return ErrorMessages.UNSUPPORTED_VALUE.format(
            param=param,
            value=value,
            allowed=sorted(allowed)
        )

    @staticmethod
    def format_must_be_one_of(param: str, allowed: Set[str], value: str = None) -> str:
        """Format must be one of error message."""
        msg = f"{param} must be one of {sorted(allowed)}"
        if value:
            msg += f", got: '{value}'"
        return msg

    @staticmethod
    def format_must_be_positive(param: str) -> str:
        """Format must be positive error message."""
        return f"{param} must be positive"

    @staticmethod
    def format_state_not_set(param: str, operation: str) -> str:
        """Format state not set error message."""
        return ErrorMessages.STATE_NOT_SET.format(
            param=param,
            operation=operation
        )

    @staticmethod
    def format_readout_observables_invalid(measurement_basis: str, allowed: Set[str]) -> str:
        """Format readout observables validation error."""
        return (
            f"readout_observables contained no supported entries for "
            f"measurement_basis='{measurement_basis}'. "
            f"Allowed values: {sorted(allowed)}"
        )
