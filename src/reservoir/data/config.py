from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class DataGenerationConfig:
    """Lightweight configuration for dataset generators."""

    name: str = "sine_wave"
    time_steps: int = 1000
    dt: float = 0.01
    noise_level: float = 0.0
    n_input: Optional[int] = None
    n_output: Optional[int] = None
    warmup_steps: Optional[int] = None
    params: Optional[dict] = None

    def get_param(self, key: str, default=None):
        if not self.params:
            return default
        return self.params.get(key, default)
