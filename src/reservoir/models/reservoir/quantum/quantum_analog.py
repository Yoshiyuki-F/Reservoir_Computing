"""
Analog (continuous-time) quantum reservoir computer built on QuTiP.

This backend simulates Rydberg-style detuning driven dynamics and exposes
the same training/prediction surface as the existing gate-based reservoir.
Input features are encoded via local detuning patterns and the readout is
trained with ridge regression on expectation values of Pauli observables.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import jax.numpy as jnp
import qutip as qt

from reservoir.models.reservoir.base import BaseReservoirComputer
from reservoir.models.reservoir.classical.config import AnalogQuantumReservoirConfig, parse_ridge_lambdas
from reservoir.components import RidgeReadoutNumpy
from reservoir.models.reservoir.training import train_reservoir, predict_reservoir

# --------------------------------------------------------------------------- #
# Helper dataclasses / utilities                                             #
# --------------------------------------------------------------------------- #


@dataclass
class _NormalizationStats:
    """Simple container for min/max statistics used in detuning scaling."""

    mean: float
    std: float

    def normalize(self, value: float, scale: float) -> float:
        """Normalize value with z-score and clip to [-1, 1]."""
        if self.std <= 1e-9:
            z = 0.0
        else:
            z = (value - self.mean) / self.std
        return float(np.clip(z * scale, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Analog quantum reservoir implementation                                    #
# --------------------------------------------------------------------------- #


class AnalogQuantumReservoir(BaseReservoirComputer):
    """
    Continuous-time quantum reservoir based on driven Rydberg interactions.

    The system evolves under a Hamiltonian with global and local detuning
    terms; observables ⟨Xᵢ⟩, ⟨Yᵢ⟩, ⟨Zᵢ⟩, and ⟨ZᵢZⱼ⟩ provide readout features.
    """

    SUPPORTED_ENCODING = {"detuning"}
    SUPPORTED_MEASUREMENTS = {"multi-pauli"}
    SUPPORTED_INPUT_MODES = {"scalar", "sequence", "block"}
    SUPPORTED_STATE_AGG = {"last", "mean", "last_mean", "mts"}

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        # Create config object - this performs all validation
        cfg = AnalogQuantumReservoirConfig(**config)

        self.backend = "qutip"
        self.config = cfg

        # Extract validated parameters
        params = cfg.params

        self.n_qubits: int = params["n_qubits"]

        # Handle positions
        self.positions: Optional[np.ndarray] = None
        raw_positions = params.get("positions")
        if raw_positions is not None:
            self.positions = np.asarray(raw_positions, dtype=np.float64)

        self.C6: float = float(params["C6"])
        self.Omega: float = float(params["Omega"])
        self.Delta_g: float = float(params["Delta_g"])
        self.Delta_l: float = float(params["Delta_l"])
        self.t_final: float = float(params["t_final"])
        self.dt: float = float(params["dt"])

        self.encoding_scheme: str = str(params["encoding_scheme"]).lower()
        self.measurement_basis: str = str(params["measurement_basis"]).lower()

        # Process readout_observables (de-duplicate)
        raw_readouts = params["readout_observables"]
        readout_seq: List[str] = []
        for entry in raw_readouts:
            name = str(entry).upper()
            if name not in readout_seq:
                readout_seq.append(name)
        self.readout_observables: Tuple[str, ...] = tuple(readout_seq)

        self.state_aggregation: str = str(params["state_aggregation"]).lower()
        self.reupload_layers: int = params["reupload_layers"]
        self.input_mode: str = str(params["input_mode"]).lower()
        self.detuning_scale: float = float(params["detuning_scale"])
        self.random_seed: int = int(params.get("random_seed", 42))

        # Parse ridge_lambdas
        self.ridge_lambdas: Sequence[float] = parse_ridge_lambdas(params)
        self._readout = RidgeReadoutNumpy()
        self._washout_steps = 0

        # Training state
        self.W_out: Optional[np.ndarray] = None
        self.best_ridge_lambda: Optional[float] = None
        self.ridge_search_log: List[Dict[str, float]] = []
        self.classification_mode: bool = False
        self.num_classes: Optional[int] = None
        self.last_training_mse: Optional[float] = None
        self.readout_logs: Optional[Any] = None
        self._washout_steps: int = 0
        self._readout = RidgeReadoutNumpy()
        self.readout_logs: Optional[Any] = None

        # Minimal interface compliance (unified reservoir API)
        self._washout_steps: int = 0

        # Normalization statistics (fit during train)
        self._norm_stats: Optional[_NormalizationStats] = None
        self.n_input_: Optional[int] = None
        self.feature_dim_: Optional[int] = None

        # Feature normalization state for design matrix
        self._feature_mu_: Optional[np.ndarray] = None
        self._feature_sigma_: Optional[np.ndarray] = None
        self._feature_keep_mask_: Optional[np.ndarray] = None

        # Random generator
        self.rng = np.random.default_rng(self.random_seed)

        # Build qubit operators and static terms
        self._ensure_positions()
        self.V_matrix = self._build_rydberg_couplings()
        self.sx_ops, self.sy_ops, self.sz_ops, self.n_ops = self._build_single_qubit_ops()
        self.H_int = self._build_entangling_static_term()

        self.H_drive_op = self.sx_ops[0] * 0
        for op in self.sx_ops:
            self.H_drive_op += 0.5 * op

        self.H_global_detuning = self.n_ops[0] * 0
        for op in self.n_ops:
            self.H_global_detuning -= op
        self.local_detuning_ops = [-op for op in self.n_ops]

        self.psi0 = qt.tensor([qt.basis(2, 0)] * self.n_qubits)
        self.tlist = np.arange(0.0, self.t_final, self.dt, dtype=np.float64)
        if self.tlist.size < 2:
            self.tlist = np.linspace(0.0, self.t_final, num=max(int(self.t_final / self.dt), 2))

        # Precompute ZZ tensor products
        self.zz_pairs: Dict[Tuple[int, int], qt.Qobj] = {}
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                self.zz_pairs[(i, j)] = self.sz_ops[i] * self.sz_ops[j]

        self.base_feature_dim = self._compute_base_feature_dim()
        if self.state_aggregation in {"last_mean", "mts"}:
            self.single_feature_dim = 2 * self.base_feature_dim
        else:
            self.single_feature_dim = self.base_feature_dim

    # ------------------------------------------------------------------ #
    # Operator construction                                              #
    # ------------------------------------------------------------------ #

    def _build_single_qubit_ops(
        self,
    ) -> Tuple[List[qt.Qobj], List[qt.Qobj], List[qt.Qobj], List[qt.Qobj]]:
        """Construct σx, σy, σz, and number operators for each site."""
        sx_list: List[qt.Qobj] = []
        sy_list: List[qt.Qobj] = []
        sz_list: List[qt.Qobj] = []
        n_list: List[qt.Qobj] = []
        eye = qt.qeye(2)

        full_eye = qt.tensor([eye] * self.n_qubits)

        for idx in range(self.n_qubits):
            op_list = [eye] * self.n_qubits

            sx_list.append(
                qt.tensor(op_list[:idx] + [qt.sigmax()] + op_list[idx + 1 :])
            )
            sy_list.append(
                qt.tensor(op_list[:idx] + [qt.sigmay()] + op_list[idx + 1 :])
            )
            sz_op = qt.tensor(op_list[:idx] + [qt.sigmaz()] + op_list[idx + 1 :])
            sz_list.append(sz_op)
            n_list.append((full_eye - sz_op) * 0.5)

        return sx_list, sy_list, sz_list, n_list

    def _build_entangling_static_term(self) -> qt.Qobj:
        """Construct the static interaction Hamiltonian."""
        H = self.n_ops[0] * 0
        for j in range(self.n_qubits):
            for k in range(j + 1, self.n_qubits):
                if self.V_matrix[j, k] == 0.0:
                    continue
                H = H + self.V_matrix[j, k] * self.n_ops[j] * self.n_ops[k]
        return H

    def _compute_base_feature_dim(self) -> int:
        """Return per-timestep feature dimensionality."""
        dim = 0
        if "X" in self.readout_observables:
            dim += self.n_qubits
        if "Y" in self.readout_observables:
            dim += self.n_qubits
        if "Z" in self.readout_observables:
            dim += self.n_qubits
        if "ZZ" in self.readout_observables:
            dim += (self.n_qubits * (self.n_qubits - 1)) // 2
        return dim

    # ------------------------------------------------------------------ #
    # Geometry helpers                                                  #
    # ------------------------------------------------------------------ #

    def _ensure_positions(self) -> None:
        """Ensure qubit positions are available; default to 1D chain."""
        if self.positions is not None:
            return

        # 1D equally spaced chain along x-axis
        coords = np.arange(self.n_qubits, dtype=np.float64).reshape(-1, 1)
        self.positions = np.hstack([coords, np.zeros((self.n_qubits, 2))])

    def _build_rydberg_couplings(self) -> np.ndarray:
        """Compute pairwise interaction strengths V_{jk}."""
        if self.positions is None:
            raise RuntimeError("Positions must be set before building couplings")

        n = self.n_qubits
        V = np.zeros((n, n), dtype=np.float64)
        for j in range(n):
            for k in range(j + 1, n):
                diff = self.positions[j] - self.positions[k]
                dist = np.linalg.norm(diff)
                if dist <= 0.0:
                    raise ValueError("Qubit positions must be distinct")
                coupling = self.C6 / (dist ** 6)
                V[j, k] = V[k, j] = coupling
        return V

    # ------------------------------------------------------------------ #
    # Hamiltonian construction                                          #
    # ------------------------------------------------------------------ #

    def _build_time_dependent_H(self, f_local: np.ndarray) -> List[Any]:
        """Create the QuTiP time-dependent Hamiltonian list."""
        if f_local.shape != (self.n_qubits,):
            raise ValueError("f_local must have shape (n_qubits,)")

        H_terms: List[Any] = []

        H_terms.append([self.H_drive_op, lambda t, args=None: self.Omega])
        H_terms.append([self.H_global_detuning, lambda t, args=None: self.Delta_g])

        for idx, coeff in enumerate(f_local):
            if abs(coeff) < 1e-12:
                continue
            scaled = float(coeff) * self.Delta_l
            op = self.local_detuning_ops[idx]
            H_terms.append([op, lambda t, args=None, c=scaled: c])

        H_terms.append(self.H_int)
        return H_terms

    # ------------------------------------------------------------------ #
    # Encoding utilities                                                #
    # ------------------------------------------------------------------ #

    def _fit_normalizer(self, data: np.ndarray) -> None:
        """Fit z-score statistics on flattened data."""
        flattened = np.asarray(data, dtype=np.float64).reshape(-1)
        self._norm_stats = _NormalizationStats(
            mean=float(np.mean(flattened)),
            std=float(np.std(flattened) + 1e-9),
        )

    def _normalize_value(self, value: float) -> float:
        if self._norm_stats is None:
            raise RuntimeError("Normalization statistics are not fitted")
        return self._norm_stats.normalize(value, self.detuning_scale)

    def _generate_detuning_layers(self, x: np.ndarray) -> List[np.ndarray]:
        """Generate detuning vectors (length n_qubits) for each reupload layer."""
        mode = self.input_mode
        values: np.ndarray

        if mode == "scalar":
            val = float(np.asarray(x).reshape(()))
            norm = self._normalize_value(val)
            return [np.full(self.n_qubits, norm, dtype=np.float64)]

        vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if mode not in {"sequence", "block"}:
            raise ValueError(f"Unsupported input_mode '{mode}'")

        total = self.n_qubits * self.reupload_layers
        if vec.size > total:
            raise ValueError(
                f"Input dimension {vec.size} exceeds capacity "
                f"{self.n_qubits}×{self.reupload_layers}"
            )

        normalized = np.zeros(total, dtype=np.float64)
        for idx, value in enumerate(vec):
            normalized[idx] = self._normalize_value(float(value))

        layers: List[np.ndarray] = []
        for layer in range(self.reupload_layers):
            start = layer * self.n_qubits
            end = start + self.n_qubits
            chunk = normalized[start:end]
            layers.append(chunk.copy())
        return layers

    # ------------------------------------------------------------------ #
    # Time evolution & measurement                                      #
    # ------------------------------------------------------------------ #

    def _evolve(self, H: List[Any], psi0: qt.Qobj) -> qt.Qobj:
        """Evolve the system under Hamiltonian H and return final state."""
        result = qt.mesolve(H, psi0, self.tlist, [], [])
        return result.states[-1]

    def _measure_features(self, state: qt.Qobj) -> np.ndarray:
        """Measure one- and two-body observables for the given state."""
        expectations: List[float] = []

        for observable in self.readout_observables:
            if observable == "X":
                for op in self.sx_ops:
                    expectations.append(np.real(qt.expect(op, state)))
            elif observable == "Y":
                for op in self.sy_ops:
                    expectations.append(np.real(qt.expect(op, state)))
            elif observable == "Z":
                for op in self.sz_ops:
                    expectations.append(np.real(qt.expect(op, state)))
            elif observable == "ZZ":
                for (i, j), op in self.zz_pairs.items():
                    expectations.append(np.real(qt.expect(op, state)))

        return np.asarray(expectations, dtype=np.float64)

    def _aggregate_features(self, features: Sequence[np.ndarray]) -> np.ndarray:
        """Aggregate a sequence of feature vectors according to policy."""
        if not features:
            return np.zeros(self.base_feature_dim, dtype=np.float64)

        stacked = np.vstack(features)
        last = stacked[-1]
        if self.state_aggregation == "last":
            return last

        mean = stacked.mean(axis=0)
        if self.state_aggregation == "mean":
            return mean
        if self.state_aggregation in {"last_mean", "mts"}:
            return np.concatenate([last, mean], axis=0)

        raise ValueError(f"Unsupported state_aggregation '{self.state_aggregation}'")

    # ------------------------------------------------------------------ #
    # Feature extraction                                                #
    # ------------------------------------------------------------------ #

    def _extract_features_scalar(self, inputs: np.ndarray) -> np.ndarray:
        features: List[np.ndarray] = []
        for value in inputs.reshape(-1):
            detunings = self._generate_detuning_layers(np.asarray(value))
            for det in detunings:
                H = self._build_time_dependent_H(det)
                state = self._evolve(H, self.psi0)
                features.append(self._measure_features(state))
        aggregated = self._aggregate_features(features)
        self.feature_dim_ = aggregated.size
        return aggregated

    def _extract_features_sequence(self, sample: np.ndarray) -> np.ndarray:
        features: List[np.ndarray] = []
        for step in np.asarray(sample):
            detunings_layers = self._generate_detuning_layers(np.asarray(step))
            for det in detunings_layers:
                H = self._build_time_dependent_H(det)
                state = self._evolve(H, self.psi0)
                features.append(self._measure_features(state))
        aggregated = self._aggregate_features(features)
        self.feature_dim_ = aggregated.size
        return aggregated

    def _build_feature_matrix(
        self,
        inputs: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        """Construct feature matrix for regression/classification."""
        if mode == "scalar":
            rows = [
                self._extract_features_scalar(np.asarray([val]))
                for val in inputs.reshape(-1)
            ]
        elif mode in {"sequence", "block"}:
            rows = [self._extract_features_sequence(sample) for sample in inputs]
        else:
            raise ValueError(f"Unsupported input_mode '{mode}'")

        return np.vstack(rows)

    # ------------------------------------------------------------------ #
    # Ridge regression utilities                                        #
    # ------------------------------------------------------------------ #

    def _prepare_design_matrix(
        self,
        feature_matrix: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        """Standardize features and filter zero-variance columns."""
        features_np = np.asarray(feature_matrix, dtype=np.float64)

        if features_np.ndim != 2:
            features_np = features_np.reshape(features_np.shape[0], -1)

        if fit or self._feature_mu_ is None or self._feature_sigma_ is None:
            mu = features_np.mean(axis=0)
            sigma = features_np.std(axis=0)
            keep = sigma >= 1e-12
            if not np.any(keep):
                keep = np.ones_like(sigma, dtype=bool)
            sigma_adj = sigma.copy()
            sigma_adj[~keep] = 1.0
            self._feature_mu_ = mu
            self._feature_sigma_ = sigma_adj
            self._feature_keep_mask_ = keep

            kept_sigma = sigma_adj[keep]
            print(
                f"[analog-qrc] feature std range (kept) -> min={float(kept_sigma.min()):.3e}, max={float(kept_sigma.max()):.3e}"
            )
        else:
            if self._feature_mu_ is None or self._feature_sigma_ is None or self._feature_keep_mask_ is None:
                raise RuntimeError("Feature scaler has not been fitted. Call train() before predict().")
            mu = self._feature_mu_
            sigma_adj = self._feature_sigma_
            keep = self._feature_keep_mask_

        centered = (features_np - mu) / sigma_adj
        centered = centered[:, keep]

        bias = np.ones((centered.shape[0], 1), dtype=np.float64)
        design = np.concatenate([centered, bias], axis=1)
        return design

    # Minimal interface compliance for unified training API ----------------
    @property
    def readout(self):
        return self._readout

    @property
    def washout_steps(self) -> int:
        return self._washout_steps

    def get_states(self, inputs):
        arr = np.asarray(inputs, dtype=np.float64)
        if self._norm_stats is None:
            self._fit_normalizer(arr)
        # Choose mode based on dimensionality
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            mode = "scalar"
        else:
            mode = self.input_mode
        features = self._build_feature_matrix(arr, mode=mode)
        return jnp.asarray(features, dtype=jnp.float64)

    def transform_states(self, raw_states, fit: bool = False):
        design = self._prepare_design_matrix(np.asarray(raw_states, dtype=np.float64), fit=fit)
        return jnp.asarray(design, dtype=jnp.float64)

    def _select_ridge_lambda(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lambdas: Sequence[float],
        classification: bool = False,
    ) -> np.ndarray:
        """Grid-search ridge parameter and return optimal weights."""
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(Y, dtype=np.float64)

        lambda_candidates = sorted(set([float(lam) for lam in lambdas if lam > 0]))

        n_samples = X_np.shape[0]
        if n_samples < 2:
            raise ValueError("Not enough samples to perform ridge regression")

        split_idx = int(0.9 * n_samples)
        split_idx = max(1, min(split_idx, n_samples - 1))

        X_train, X_val = X_np[:split_idx], X_np[split_idx:]
        y_train, y_val = y_np[:split_idx], y_np[split_idx:]

        feature_part = X_train[:, :-1]
        if feature_part.size:
            sigma = feature_part.std(axis=0)
            print(
                f"[analog-qrc] feature std range (train) -> min={float(sigma.min()):.3e}, max={float(sigma.max()):.3e}"
            )
            try:
                sv = np.linalg.svd(feature_part, compute_uv=False)
                cond_number = float(sv.max() / max(sv.min(), 1e-12))
                print(f"[analog-qrc] design matrix cond -> {cond_number:.3e}")
            except np.linalg.LinAlgError:
                print("[analog-qrc] SVD failed; skipping condition number log.")

        # Legacy ridge search removed; unified training handles readout fitting.
    # ------------------------------------------------------------------ #
    # Training / prediction (regression)                                #
    # ------------------------------------------------------------------ #

    def train(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
        return_features: bool = False,
    ) -> Optional[np.ndarray]:
        """Train the analog reservoir using the unified readout."""
        # Fit input normalizer once per training call
        self._fit_normalizer(np.asarray(input_data, dtype=np.float64))
        result = train_reservoir(
            self,
            jnp.asarray(input_data),
            jnp.asarray(target_data),
            mode="regression",
            ridge_lambdas=ridge_lambdas,
            num_classes=None,
        )
        self.last_training_mse = result.get("score")
        if return_features:
            return self.get_states(jnp.asarray(input_data))
        return None

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Generate predictions for regression data."""
        return np.array(predict_reservoir(self, jnp.asarray(input_data), mode="regression"))

    # ------------------------------------------------------------------ #
    # Classification helpers                                            #
    # ------------------------------------------------------------------ #

    def train_classification(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
        num_classes: Optional[int] = None,
        return_features: bool = False,
    ) -> None:
        """Train the reservoir for classification tasks."""
        data = np.asarray(sequences, dtype=np.float64)
        self._fit_normalizer(data)
        train_reservoir(
            self,
            jnp.asarray(sequences),
            jnp.asarray(labels),
            mode="classification",
            ridge_lambdas=ridge_lambdas,
            num_classes=num_classes,
        )
        if return_features:
            return self.get_states(jnp.asarray(sequences))
        return None

    def predict_classification(
        self,
        sequences: Optional[np.ndarray] = None,
        return_logits: bool = False,
        progress_desc: Optional[str] = None,
        progress_position: int = 0,
        precomputed_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict logits or labels for classification."""
        del progress_desc, progress_position
        if precomputed_features is None and sequences is None:
            raise ValueError("Either sequences or precomputed_features must be provided.")
        if precomputed_features is not None and sequences is not None:
            raise ValueError("Specify only one of sequences or precomputed_features.")
        inputs = precomputed_features if precomputed_features is not None else sequences
        logits = predict_reservoir(
            self,
            jnp.asarray(inputs),
            precomputed_features=jnp.asarray(precomputed_features) if precomputed_features is not None else None,
            mode="classification",
        )
        if return_logits:
            return np.array(logits)
        return np.array(jnp.argmax(logits, axis=1))

    # ------------------------------------------------------------------ #
    # Housekeeping                                                      #
    # ------------------------------------------------------------------ #

    def reset_state(self) -> None:
        """Reset training artefacts while keeping built operators."""
        super().reset_state()
        self.W_out = None
        self.classification_mode = False
        self.num_classes = None
        self.best_ridge_lambda = None
        self.ridge_search_log.clear()
        self.last_training_mse = None
        self.feature_dim_ = None
        self._norm_stats = None
        self._feature_mu_ = None
        self._feature_sigma_ = None
        self._feature_keep_mask_ = None
        if hasattr(self._readout, "weights"):
            try:
                self._readout.weights = None  # type: ignore[attr-defined]
            except Exception:
                pass

    def get_reservoir_info(self) -> Dict[str, Any]:
        """Return reservoir metadata for logging/debugging."""
        info = {
            "model_type": "quantum_analog",
            "n_qubits": self.n_qubits,
            "reupload_layers": self.reupload_layers,
            "state_aggregation": self.state_aggregation,
            "encoding_scheme": self.encoding_scheme,
            "measurement_basis": self.measurement_basis,
            "readout_observables": list(self.readout_observables),
            "feature_dim": self.feature_dim_
            if self.feature_dim_ is not None
            else (
                2 * self.base_feature_dim
                if self.state_aggregation in {"last_mean", "mts"}
                else self.base_feature_dim
            ),
            "Omega": self.Omega,
            "Delta_g": self.Delta_g,
            "Delta_l": self.Delta_l,
            "t_final": self.t_final,
            "dt": self.dt,
            "backend": self.backend,
            "trained": self.trained,
        }
        return info
