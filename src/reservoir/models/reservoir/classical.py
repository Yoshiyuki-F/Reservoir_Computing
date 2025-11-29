"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/classical.py
Classical Reservoir Computer implementation using JAX.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Iterable, Callable, cast

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax import random
from tqdm.auto import tqdm

from reservoir.utils import ensure_x64_enabled

ensure_x64_enabled()

from reservoir.components import (
    FeatureScaler,
    DesignMatrixBuilder,
    aggregate_states,
    BaseReadout,
    RidgeReadoutNumpy,
    RidgeReadoutJAX,
)
from reservoir.components.preprocess.aggregator import AggregationMode
from .utils.spectral import spectral_radius_scale
from reservoir.models.reservoir.training import train_reservoir, predict_reservoir

from .base import BaseReservoirComputer
from .config import ReservoirConfig, parse_ridge_lambdas


@lru_cache()
def _load_shared_defaults() -> Dict[str, Any]:
    """Load shared reservoir default parameters from presets/models."""
    root = Path(__file__).resolve()
    target_rel = Path("presets/models/shared_reservoir_params.json")
    for parent in root.parents:
        candidate = parent / target_rel
        if candidate.is_file():
            data = json.loads(candidate.read_text())
            return dict(data.get("params", {}))
    raise FileNotFoundError(
        f"Could not find {target_rel} relative to {root}. "
        "Ensure presets/models/shared_reservoir_params.json is present."
    )


class NNXReservoirCell(nnx.Module):
    """
    NNX Reservoir Cell used primarily for initialization.
    Actual execution uses pure JAX functions for performance.
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        spectral_radius: float,
        input_scaling: float,
        reservoir_weight_range: float,
        leaky_rate: float,
        rngs: nnx.Rngs,
    ):
        self.n_hidden = n_hidden
        self.leaky_rate = leaky_rate
        self.rngs = rngs

        self.w_in = nnx.Linear(
            n_inputs,
            n_hidden,
            use_bias=False,
            param_dtype=jnp.float64,
            rngs=self.rngs,
        )
        self.w_res = nnx.Linear(
            n_hidden,
            n_hidden,
            use_bias=False,
            param_dtype=jnp.float64,
            rngs=self.rngs,
        )

        # Initialize Input Weights
        key_in = self.rngs.params()
        w_in_init = jax.random.uniform(
            key_in,
            (n_inputs, n_hidden),
            minval=-input_scaling,
            maxval=input_scaling,
            dtype=jnp.float64,
        )
        self.w_in.kernel.value = w_in_init

        # Initialize Reservoir Weights
        key_res = self.rngs.params()
        w_res_init = jax.random.uniform(
            key_res,
            (n_hidden, n_hidden),
            minval=-reservoir_weight_range,
            maxval=reservoir_weight_range,
            dtype=jnp.float64,
        )
        w_res_scaled = spectral_radius_scale(w_res_init, spectral_radius)
        self.w_res.kernel.value = w_res_scaled


class ReservoirComputer(BaseReservoirComputer):
    """
    JAX-based Echo State Network (ESN) implementation.
    Supports both Regression and Classification via auto-detection.
    """

    def __init__(
        self,
        config: Sequence[Dict[str, Any]],
        backend: Optional[str] = None,
        readout: Optional[BaseReadout] = None,
    ):
        super().__init__()

        # --- Configuration Loading ---
        merged: Dict[str, Any] = _load_shared_defaults().copy()
        config_sequence: Iterable[Dict[str, Any]] = [config] if isinstance(config, dict) else config
        for cfg in config_sequence:
            cfg_dict = dict(cfg)
            merged.update({k: v for k, v in cfg_dict.items() if k not in {'name', 'description', 'params'}})
            params = cfg_dict.get('params', {}) or {}
            merged.update(params)

        cfg = ReservoirConfig(**merged)
        self.config = cfg
        params = cfg.params

        # --- Parameter Extraction ---
        self.n_inputs: int = params['n_inputs']
        self.n_hidden_layer: int = params['n_hidden_layer']
        self.n_outputs: int = params['n_outputs']
        self.spectral_radius: float = float(params['spectral_radius'])
        self.input_scaling: float = float(params['input_scaling'])
        self.noise_level: float = float(params['noise_level'])
        self.alpha: float = float(params['alpha'])
        self.reservoir_weight_range: float = float(params['reservoir_weight_range'])
        self.sparsity: float = float(params['sparsity'])
        self.input_bias: float = float(params['input_bias'])
        self.encode_batch_size: int = max(1, int(params.get('encode_batch_size', 1024)))
        random_seed: int = int(params['random_seed'])
        state_agg = str(params.get('state_aggregation', 'last')).lower()
        self.state_aggregation: AggregationMode = cast(AggregationMode, state_agg)
        self.use_preprocessing: bool = bool(params.get('use_preprocessing', True))

        self.backend = backend
        self.initial_random_seed = random_seed

        # --- Initialization ---
        self.rngs = nnx.Rngs(params=random_seed, noise=random_seed)
        # Store a static key for pure JAX operations (vmap/scan)
        self.key = random.PRNGKey(random_seed)

        # Build Cell and Extract Weights for JAX Functional Use
        self.cell = self._build_cell()
        self.W_in = jnp.asarray(self.cell.w_in.kernel.value.T, dtype=jnp.float64)
        self.W_res = jnp.asarray(self.cell.w_res.kernel.value, dtype=jnp.float64)

        # State Variables
        self.W_out = None
        self.best_ridge_lambda: Optional[float] = None
        self.ridge_search_log: list[Dict[str, float]] = []
        self.last_training_mse: Optional[float] = None
        self.classification_mode: bool = False
        self.num_classes: Optional[int] = None
        self._washout_steps: int = 3
        self.ridge_lambdas: Sequence[float] = parse_ridge_lambdas(params)

        # Readout Components
        design_cfg = {
            "poly_mode": params.get("poly_mode", "square"),
            "degree": int(params.get("poly_degree", 2)),
            "include_bias": True,
            "std_threshold": float(params.get("std_threshold", 1e-3)),
        }
        self._design_cfg = design_cfg
        self.scaler = FeatureScaler()
        self.design_builder = DesignMatrixBuilder(**design_cfg)
        self._readout_cv = params.get("readout_cv", "holdout")
        self._readout_n_folds = int(params.get("readout_n_folds", 5))
        self._external_readout = readout
        if readout is not None:
            self._readout = readout
        elif (backend or "").lower() == "gpu":
            self._readout = RidgeReadoutJAX()
        else:
            self._readout = RidgeReadoutNumpy(
                default_cv=self._readout_cv,
                default_n_folds=self._readout_n_folds,
            )
        self.last_training_score: Optional[float] = None
        self.last_training_score_name: Optional[str] = None

    # Interface adapters -------------------------------------------------
    @property
    def readout(self) -> BaseReadout:
        return self._readout

    @property
    def washout_steps(self) -> int:
        return self._washout_steps

    def _build_cell(self) -> NNXReservoirCell:
        cell_rngs = self.rngs.fork()
        return NNXReservoirCell(
            n_inputs=self.n_inputs,
            n_hidden=self.n_hidden_layer,
            spectral_radius=self.spectral_radius,
            input_scaling=self.input_scaling,
            reservoir_weight_range=self.reservoir_weight_range,
            leaky_rate=self.alpha,
            rngs=cell_rngs,
        )

    def _run_batch_sequences(self, batch_sequences: jnp.ndarray) -> jnp.ndarray:
        """
        Pure JAX implementation of reservoir dynamics using vmap + scan.
        This handles RNG splitting explicitly to avoid NNX TraceContextErrors during JIT.
        """
        batch_sequences = jnp.asarray(batch_sequences, dtype=jnp.float64)

        # Weights as static JAX arrays
        W_in = jnp.asarray(self.W_in, dtype=jnp.float64)
        W_res = jnp.asarray(self.W_res, dtype=jnp.float64)

        # Pre-project inputs: (B, T, I) @ (I, H) -> (B, T, H)
        projected = jnp.dot(batch_sequences, W_in.T)

        # Prepare RNG keys: One per sequence in the batch
        keys = random.split(self.key, batch_sequences.shape[0] + 1)
        self.key = keys[0] # Update object state key
        seq_keys = keys[1:]

        # --- Scan Function for a Single Sequence ---
        def run_single_sequence(seq_proj: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
            def step(carry, inp):
                state, k = carry
                k, sub = random.split(k)

                # Explicit noise generation
                noise = random.normal(sub, (self.n_hidden_layer,), dtype=jnp.float64) * self.noise_level

                # ESN State Update: h(t) = (1-a)*h(t-1) + a*tanh( W_res@h(t-1) + W_in@u(t) + noise )
                # Note: 'inp' here is already W_in@u(t)
                res_term = jnp.dot(W_res, state)
                pre_activation = res_term + inp + noise
                new_state = (1.0 - self.alpha) * state + self.alpha * jnp.tanh(pre_activation)

                return (new_state, k), new_state

            init_state = jnp.zeros((self.n_hidden_layer,), dtype=jnp.float64)
            # Scan returns (final_carry, all_states)
            (_, _), states = jax.lax.scan(step, (init_state, key), seq_proj)
            return states

        # --- Parallelize over Batch ---
        batch_states = jax.vmap(run_single_sequence)(projected, seq_keys)
        return batch_states

    def get_states(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Return raw states or aggregated features depending on input rank."""
        arr = jnp.asarray(inputs, dtype=jnp.float64)
        if arr.ndim == 3:
            # Batched sequences -> aggregated features per sequence
            return self.encode_batch(arr)
        # Single sequence -> full state trajectory
        return self.run_hidden_layer(arr)

    def transform_states(self, raw_states: jnp.ndarray, fit: bool = False) -> jnp.ndarray:
        """Transform raw states into design features (scaler + polynomial expansion)."""
        data = jnp.asarray(raw_states, dtype=jnp.float64)
        if not self.use_preprocessing:
            bias = jnp.ones((data.shape[0], 1), dtype=data.dtype)
            return jnp.concatenate([data, bias], axis=1)
        if fit:
            normalized = self.scaler.fit_transform(data)
            return self.design_builder.fit_transform(normalized)
        normalized = self.scaler.transform(data)
        return self.design_builder.transform(normalized)

    def run_hidden_layer(self, input_sequence: jnp.ndarray) -> jnp.ndarray:
        """Run reservoir on a single sequence (wrapper)."""
        seq_batch = jnp.asarray(input_sequence, dtype=jnp.float64)[None, ...]
        states_batch = self._run_batch_sequences(seq_batch)
        return states_batch[0]

    def encode_batch(self, batch_sequences: jnp.ndarray) -> jnp.ndarray:
        """Public alias for encoding sequences, often used by quantum/hybrid pipelines."""
        return self._encode_sequences(batch_sequences, desc="Encoding sequences")

    def _encode_sequences(
        self,
        sequences: jnp.ndarray,
        desc: Optional[str] = None,
        *,
        leave: bool = False,
        batch_size: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Runs the reservoir on batches of sequences and applies state aggregation.
        """
        sequences = jnp.asarray(sequences, dtype=jnp.float64)
        total = sequences.shape[0]
        if total == 0:
            raise ValueError("Cannot encode empty sequence batch.")

        batch = int(batch_size or self.encode_batch_size)
        batch = max(1, batch)

        # Aggregation function (vmapped)
        agg_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda st: aggregate_states(st, self.state_aggregation)
        aggregate_batch_fn = jax.vmap(agg_fn, in_axes=0)

        features_list = []
        iterator = range(0, total, batch)
        if desc:
            iterator = tqdm(iterator, desc=desc, leave=leave, unit="batch")

        for start in iterator:
            end = min(start + batch, total)
            batch_sequences = sequences[start:end]

            # 1. Get full time-series states (B, T, H)
            batch_states = self._run_batch_sequences(batch_sequences)
            # 2. Aggregate to features (B, F)
            batch_features = aggregate_batch_fn(batch_states)

            features_list.append(batch_features)

        concatenated = jnp.concatenate(features_list, axis=0)
        return concatenated

    def train(
        self,
        input_data: jnp.ndarray,
        target_data: jnp.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Unified training interface.
        Automatically detects Classification (integer labels) vs Regression (continuous values).
        """
        # Auto-detect Classification: 1D array of integers
        is_classification_labels = (
            target_data.ndim == 1 and
            (jnp.issubdtype(target_data.dtype, jnp.integer) or jnp.issubdtype(target_data.dtype, jnp.bool_))
        )

        mode = "classification" if is_classification_labels else "regression"
        num_classes = None
        if mode == "classification":
            num_classes = int(self.n_outputs) if self.n_outputs > 1 else int(jnp.max(target_data) + 1)

        return train_reservoir(
            self,
            input_data,
            target_data,
            mode=mode,  # type: ignore[arg-type]
            ridge_lambdas=ridge_lambdas,
            num_classes=num_classes,
        )

    def predict(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """
        Predicts outputs.
        Returns Logits (for classification) or Values (for regression) based on training mode.
        """
        mode = "classification" if self.classification_mode else "regression"
        return predict_reservoir(self, input_data, mode=mode)  # type: ignore[arg-type]

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        """
        Evaluates the model.
        Returns Accuracy (classification) or MSE/MAE (regression).
        """
        if self.classification_mode:
            # --- Classification Evaluation ---
            logits = self.predict(X)
            pred_labels = jnp.argmax(logits, axis=1)

            y_labels = y.astype(jnp.int32)
            # Handle one-hot targets if passed by mistake, though expected 1D labels
            if y_labels.ndim > 1:
                y_labels = jnp.argmax(y_labels, axis=1)

            accuracy = float(jnp.mean(pred_labels == y_labels))
            return {"accuracy": accuracy}
        else:
            # --- Regression Evaluation ---
            return super().evaluate(X, y)

    def reset_state(self) -> None:
        """Full reset of the reservoir parameters and training state."""
        super().reset_state()
        self.W_out = None
        self.best_ridge_lambda = None
        self.ridge_search_log = []
        self.last_training_mse = None
        self.last_training_score = None
        self.last_training_score_name = None
        self.classification_mode = False
        self.num_classes = None
        self.readout_logs = None

        # Reset Preprocessing
        self.scaler = FeatureScaler()
        self.design_builder = DesignMatrixBuilder(**self._design_cfg)

        # Reset Readout
        if self._external_readout is None:
            if (self.backend or "").lower() == "gpu":
                self._readout = RidgeReadoutJAX()
            else:
                self._readout = RidgeReadoutNumpy(
                    default_cv=self._readout_cv,
                    default_n_folds=self._readout_n_folds,
                )
        else:
            self._readout = self._external_readout
            if hasattr(self._readout, "weights"):
                setattr(self._readout, "weights", None)

        # Reset Weights (New Seed)
        self.rngs = nnx.Rngs(params=self.initial_random_seed, noise=self.initial_random_seed)
        self.key = random.PRNGKey(self.initial_random_seed)

        self.cell = self._build_cell()
        self.W_in = jnp.asarray(self.cell.w_in.kernel.value.T, dtype=jnp.float64)
        self.W_res = jnp.asarray(self.cell.w_res.kernel.value, dtype=jnp.float64)

    def get_reservoir_info(self) -> Dict[str, Any]:
        return {
            **(self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config),
            "backend": self.backend,
            "trained": self.trained,
            "classification_mode": self.classification_mode
        }
