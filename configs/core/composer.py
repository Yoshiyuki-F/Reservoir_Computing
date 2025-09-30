"""Model-agnostic configuration composition utilities."""

import json
from pathlib import Path
from typing import Dict, Any, Union
from dataclasses import dataclass


@dataclass
class ComposedConfig:
    """Composed configuration from multiple config files."""
    dataset: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    visualization: Dict[str, Any]  # From experiment config
    experiment_name: str
    experiment_description: str


class ConfigComposer:
    """Composes configurations from modular config files."""

    def __init__(self, config_root: Union[str, Path] = "configs"):
        """Initialize config composer.

        Args:
            config_root: Root directory containing config files
        """
        self.config_root = Path(config_root)

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a single config file.

        Args:
            config_name: Name of config file (without .json extension)

        Returns:
            Config dictionary
        """
        config_path = self.config_root / f"{config_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            return json.load(f)

    def load_from_category(self, category: str, name: str) -> Dict[str, Any]:
        """Load config from a specific category directory.

        Args:
            category: Category directory (datasets, models, training, etc.)
            name: Config name within category

        Returns:
            Config dictionary
        """
        config_path = self.config_root / category / f"{name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            return json.load(f)

    def compose_experiment(self, experiment_config: Union[str, Dict[str, Any]]) -> ComposedConfig:
        """Compose a complete experiment configuration.

        Args:
            experiment_config: Either experiment config name or config dictionary

        Returns:
            Composed configuration
        """
        if isinstance(experiment_config, str):
            exp_config = self.load_from_category("experiments", experiment_config)
        else:
            exp_config = experiment_config

        # Load each component
        dataset_config = self.load_from_category("datasets", exp_config["dataset"])
        model_config = self.load_from_category("models", exp_config["model"])
        training_config = self.load_from_category("training", exp_config["training"])

        # Visualization is now in experiment config
        visualization_config = exp_config.get("visualization", {})

        return ComposedConfig(
            dataset=dataset_config,
            model=model_config,
            training=training_config,
            visualization=visualization_config,
            experiment_name=exp_config["name"],
            experiment_description=exp_config["description"]
        )

    def compose_legacy_format(self, composed: ComposedConfig) -> Dict[str, Any]:
        """Convert composed config back to legacy format for compatibility.

        Args:
            composed: Composed configuration

        Returns:
            Legacy format configuration dictionary
        """
        # Extract model type to determine which model config to use
        model_name = composed.model["name"]

        legacy_config = {
            "data_generation": {
                k: v for k, v in composed.dataset.items()
                if k != "description"  # Filter out description for compatibility
            }
        }

        # Add model config to appropriate section based on model type
        model_params = composed.model.get("params", {})
        model_type = composed.model.get("model_type", composed.model.get("type", "reservoir"))

        if "quantum" in model_name or model_type == "quantum":
            # Quantum model configuration
            legacy_config["quantum_reservoir"] = model_params.copy()
            # Add minimal reservoir section for compatibility
            legacy_config["reservoir"] = {
                "n_inputs": model_params.get("n_inputs", 1),
                "n_outputs": model_params.get("n_outputs", 1)
            }
        else:
            # Classical reservoir or other model configuration
            legacy_config["reservoir"] = model_params.copy()
            # Add None quantum section for compatibility
            legacy_config["quantum_reservoir"] = None

        # Add generic model config for the new dynamic system
        legacy_config["model"] = {
            "name": model_name,
            "model_type": model_type,
            "params": model_params.copy()
        }

        # Add training config
        legacy_config["training"] = {
            k: v for k, v in composed.training.items()
            if k not in ["description", "preprocessing", "train_size"]  # Filter out invalid fields
        }
        # Ensure training has required name field
        if "name" not in legacy_config["training"]:
            legacy_config["training"]["name"] = composed.training.get("name", "standard")

        # Add preprocessing if present
        if "preprocessing" in composed.training:
            legacy_config["preprocessing"] = composed.training["preprocessing"]

        # Add visualization config as demo (from experiment) with auto-generated title and filename
        demo_config = composed.visualization.copy()

        # Auto-generate title and filename
        dataset_name = composed.dataset["name"]
        is_quantum = "quantum" in model_name

        # Create human-readable dataset name
        dataset_display_names = {
            "sine_wave": "Sine Wave",
            "lorenz": "Lorenz Attractor",
            "mackey_glass": "Mackey-Glass"
        }

        display_name = dataset_display_names.get(dataset_name, dataset_name.replace("_", " ").title())

        # Auto-generate title: (Quantum) + Dataset + Prediction
        quantum_prefix = "Quantum " if is_quantum else ""
        auto_title = f"{quantum_prefix}{display_name} Prediction"

        # Auto-generate filename: dataset_name + (quantum) + _prediction.png
        model_type = "_quantum" if is_quantum else ""
        auto_filename = f"{dataset_name}{model_type}_prediction.png"

        demo_config["title"] = auto_title
        demo_config["filename"] = auto_filename
        legacy_config["demo"] = demo_config

        return legacy_config


def load_experiment_config(experiment_name: str, config_root: str = "configs") -> Dict[str, Any]:
    """Convenience function to load and compose experiment config in legacy format.

    Args:
        experiment_name: Name of experiment config
        config_root: Root directory containing config files

    Returns:
        Legacy format configuration dictionary
    """
    composer = ConfigComposer(config_root)
    composed = composer.compose_experiment(experiment_name)
    return composer.compose_legacy_format(composed)