import numpy as np
import pytest

from reservoir.data.config import MackeyGlassConfig
from reservoir.data.generators import generate_mackey_glass_data


@pytest.mark.parametrize("washup_a, washup_b", [(0, 1), (0, 3)])
def test_mackey_glass_warmup_changes_series(washup_a: int, washup_b: int):
    """Washup length (in Lyapunov time) should shift the Mackey-Glass trajectory."""

    base_params = dict(
        n_input=1,
        n_output=1,
        dt=1,  # dt=1 so steps_per_lt = lyapunov_time_unit / dt = 166
        noise_level=0.0,
        seed=0,
        tau=17,
        beta=0.2,
        gamma=0.1,
        n=10,
        downsample=1,
        lyapunov_time_unit=166.6,  # Mackey-Glass LT
        train_lt=1,
        val_lt=0,
        test_lt=0,
    )

    config_a = MackeyGlassConfig(**base_params, washup_lt=washup_a)
    config_b = MackeyGlassConfig(**base_params, washup_lt=washup_b)

    series_a, _ = generate_mackey_glass_data(config_a)
    series_b, _ = generate_mackey_glass_data(config_b)

    # Ensure same shape (train_lt=1 -> int(166.6/1)=166 steps + 1 - 1 = 166)
    assert series_a.shape == series_b.shape == (166, 1)

    # The warmup shift should give us different trajectories.
    assert not np.allclose(
        np.asarray(series_a).flatten(),
        np.asarray(series_b).flatten(),
        atol=1e-6,
    )
