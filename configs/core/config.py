"""
Model-agnostic configuration system for ML experiments.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class DataGenerationConfig(BaseModel):
    """Time series data generation configuration.

    Contains parameters needed for generating various types of time series data.
    Additional dataset-specific parameters are stored in the `params` field.
    """
    name: str = Field(..., description="Dataset name ('sine_wave', 'lorenz', 'mackey_glass')")
    time_steps: int = Field(..., gt=0, description="Length of time series")
    dt: float = Field(..., gt=0, description="Time step size")
    noise_level: float = Field(0.0, ge=0, description="Noise level to add to data")
    use_dimensions: Optional[List[int]] = Field(None, description="Which dimensions to use (None = all)")
    warmup_steps: Optional[int] = Field(None, ge=0, description="Number of transient steps to discard before returning data")
    n_input: Optional[int] = Field(None, ge=1, description="Input feature dimension")
    n_output: Optional[int] = Field(None, ge=1, description="Output/target dimension")
    params: Dict[str, Any] = Field(default_factory=dict, description="Dataset-specific parameters")

    def get_param(self, param_name: str, default=None):
        """Get a parameter from the params dictionary."""
        return self.params.get(param_name, default)

    @model_validator(mode="after")
    def _validate_mnist_dimensions(self) -> 'DataGenerationConfig':
        if self.name == "mnist":
            if self.n_input is None or self.time_steps is None:
                raise ValueError("MNIST dataset requires n_input and time_steps to be set")
            total = self.n_input * self.time_steps
            if total != 28 * 28:
                raise ValueError(
                    f"MNIST dataset requires n_input * time_steps == 784, got {self.n_input} * {self.time_steps} = {total}"
                )
        return self

    model_config = ConfigDict(extra="forbid")


class TrainingConfig(BaseModel):
    """Model training configuration."""
    name: str = Field(..., description="Training configuration name")
    task_type: Literal["classification", "timeseries"] = Field(
        "timeseries",
        description="Primary task type handled by the pipeline"
    )
    train_size: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Optional train split fraction (timeseries default 0.8)"
    )
    val_size: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional validation split fraction (classification only)",
    )
    ridge_lambdas: Optional[List[float]] = Field(
        None,
        description="Candidate ridge regularization strengths for grid search (log scale recommended)"
    )
    state_aggregation: Optional[str] = Field(
        None,
        description="Optional override for reservoir state aggregation mode",
    )
    preprocessing: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional preprocessing overrides specific to this training profile",
    )
    readout: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional readout configuration overrides specific to this training profile",
    )
    learning_rate: float = Field(
        ...,
        gt=0.0,
        description="Base learning rate for gradient-based models (required when used)",
    )
    batch_size: int = Field(
        ...,
        gt=0,
        description="Mini-batch size for training (required when used)",
    )
    num_epochs: int = Field(
        ...,
        gt=0,
        description="Number of training epochs (required when used)",
    )

    model_config = ConfigDict(extra="forbid")


class PreprocessingConfig(BaseModel):
    """Data preprocessing configuration."""
    normalize: bool = Field(True, description="Whether to normalize data")
    standardize: bool = Field(False, description="Whether to standardize data")
    sequence_length: Optional[int] = Field(None, gt=0, description="Sequence length for windowing")

    model_config = ConfigDict(extra="forbid")


class DemoConfig(BaseModel):
    """Visualization and demo configuration."""
    title: str = Field(..., description="Title for the experiment")
    filename: str = Field(..., description="Output filename for results")
    show_training: bool = Field(False, description="Whether to show training data in plots")
    y_axis_label: str = Field("Value", description="Label to display on the plot y-axis")
    add_test_zoom: bool = Field(False, description="Include zoomed-in subplot of test region")
    zoom_range: Optional[List[int]] = Field(
        None,
        description="Optional [start, end] indices (in time steps) for the test zoom subplot"
    )

    model_config = ConfigDict(extra="forbid")


class ModelConfig(BaseModel):
    """Base model configuration.

    This is a flexible base class that can hold any model-specific parameters.
    Specific model types can extend this or define their own structure.
    """
    name: str = Field(..., description="Model configuration name")
    model_type: str = Field(..., description="Type of model (e.g., 'reservoir', 'quantum', 'ffn')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility


class ExperimentConfig(BaseModel):
    """Complete experiment configuration that works with any model type."""
    data_generation: DataGenerationConfig
    model: ModelConfig  # Generic model config
    training: TrainingConfig
    demo: DemoConfig
    preprocessing: Optional[PreprocessingConfig] = Field(None, description="Data preprocessing configuration")

    # Legacy compatibility fields - these will be populated dynamically based on model type
    reservoir: Optional[Dict[str, Any]] = Field(None, description="Legacy reservoir config")
    quantum_reservoir: Optional[Dict[str, Any]] = Field(None, description="Legacy quantum config")

    def get_data_params(self) -> Dict[str, Any]:
        """Get data generation parameters as dictionary"""
        return self.data_generation.model_dump()

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)

    model_config = ConfigDict(extra="forbid")
