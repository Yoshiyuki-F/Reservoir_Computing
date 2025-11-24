"""Naming and logging helpers for experiment outputs."""

from math import comb
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core_lib.core import ExperimentConfig


def resolve_experiment_naming(
    demo_config: ExperimentConfig,
    rc: Any,
    reservoir_info: Dict[str, Any],
    *,
    dataset_name: str,
    model_type: str,
    quantum_mode: bool,
    is_quantum_model: bool,
    raw_training: bool,
    n_hidden_layer: Optional[int],
) -> Tuple[str, str]:
    """Resolve output filename and plot title for an experiment."""

    resolved_filename = demo_config.demo.filename
    filename_suffix_parts = []
    if quantum_mode or "quantum" in model_type:
        suffix = Path(resolved_filename).suffix or ".png"
        resolved_filename = f"{dataset_name}_gq_prediction{suffix}"

    if raw_training:
        filename_suffix_parts.append("raw")

    if n_hidden_layer is not None and not is_quantum_model:
        filename_suffix_parts.append(f"nr{n_hidden_layer}")

    plot_title = demo_config.demo.title

    info_dict = reservoir_info if isinstance(reservoir_info, dict) else {}
    if quantum_mode or "quantum" in model_type:
        qubit_candidates: list[Any] = []
        depth_candidates: list[Any] = []

        qubit_candidates.append(info_dict.get("n_qubits"))
        depth_candidates.append(info_dict.get("circuit_depth"))

        if hasattr(rc, "n_qubits"):
            qubit_candidates.append(getattr(rc, "n_qubits"))
        if hasattr(rc, "circuit_depth"):
            depth_candidates.append(getattr(rc, "circuit_depth"))

        quantum_cfg = getattr(demo_config, "quantum_reservoir", None)
        if isinstance(quantum_cfg, dict):
            qubit_candidates.append(quantum_cfg.get("n_qubits"))
            depth_candidates.append(quantum_cfg.get("circuit_depth"))

        n_qubits: Optional[int] = None
        for candidate in qubit_candidates:
            if candidate is None:
                continue
            try:
                n_qubits = int(candidate)
                break
            except (TypeError, ValueError):
                continue

        circuit_depth: Optional[int] = None
        for candidate in depth_candidates:
            if candidate is None:
                continue
            try:
                circuit_depth = int(candidate)
                break
            except (TypeError, ValueError):
                continue

        readout_features = info_dict.get("readout_feature_dim")
        readout_observables = info_dict.get("readout_observables")
        state_agg = str(info_dict.get("state_aggregation", "")).lower()

        if n_qubits is not None and readout_observables:
            components: list[str] = []
            calculated_dim = 0
            for observable in readout_observables:
                obs = str(observable).upper()
                if obs in {"X", "Y", "Z"}:
                    calculated_dim += n_qubits
                    components.append(f"{n_qubits} {obs}")
                elif obs == "ZZ":
                    pairs = comb(int(n_qubits), 2)
                    calculated_dim += pairs
                    components.append(f"{pairs} ZZ")

            base_dim = readout_features or calculated_dim
            components_str = " + ".join(components) if components else f"{base_dim}"

            aggregated_dim = info_dict.get("feature_dim", base_dim)
            agg_note = ""
            if aggregated_dim != base_dim:
                agg_note = f"; after '{state_agg}' aggregation → {aggregated_dim}"

            print(
                f"🧮 Quantum feature dimension: {base_dim} ({components_str}){agg_note}"
            )
        elif n_qubits is not None and readout_features:
            print(f"🧮 Quantum feature dimension: {readout_features}")

        if n_qubits is not None or circuit_depth is not None:
            filename_path = Path(resolved_filename)
            suffix = filename_path.suffix or ""
            stem = filename_path.stem
            if n_qubits is not None and circuit_depth is not None:
                resolved_filename = f"{stem}_{n_qubits}_{circuit_depth}{suffix}"
            elif n_qubits is not None:
                resolved_filename = f"{stem}_{n_qubits}{suffix}"
            elif circuit_depth is not None:
                resolved_filename = f"{stem}_{circuit_depth}{suffix}"

    if filename_suffix_parts:
        filename_path = Path(resolved_filename)
        suffix = filename_path.suffix or ""
        stem = filename_path.stem
        suffix_segment = "_".join(filename_suffix_parts)
        resolved_filename = f"{stem}_{suffix_segment}{suffix}"

    dataset_dir = Path(dataset_name).name
    base_dir = Path("outputs") / dataset_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = base_dir / resolved_filename

    return str(resolved_path), plot_title
