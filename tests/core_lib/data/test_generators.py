import numpy as np
import pytest

from reservoir.core.config import DataGenerationConfig
from reservoir.data.generators import generate_mackey_glass_data


@pytest.mark.parametrize("warmup_a, warmup_b", [(0, 100), (0, 500)])
def test_mackey_glass_warmup_changes_series(warmup_a: int, warmup_b: int):
    """Warmup length should shift the Mackey-Glass trajectory."""

    base_params = dict(
        name="mackey_glass",
        time_steps=200,
        dt=0.1,
        noise_level=0.0,
        params={
            "tau": 17,
            "beta": 0.2,
            "gamma": 0.1,
            "n": 10,
            "initial_value": 1.2,
        },
    )

    config_a = DataGenerationConfig(**base_params, warmup_steps=warmup_a)
    config_b = DataGenerationConfig(**base_params, warmup_steps=warmup_b)

    series_a, _ = generate_mackey_glass_data(config_a)
    series_b, _ = generate_mackey_glass_data(config_b)

    # Ensure same shape and deterministic ordering
    assert series_a.shape == series_b.shape == (199, 1)

    # The warmup shift should give us different trajectories.
    assert not np.allclose(
        np.asarray(series_a).flatten(),
        np.asarray(series_b).flatten(),
        atol=1e-6,
    )
