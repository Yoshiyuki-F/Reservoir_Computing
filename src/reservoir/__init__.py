# Package init
# Enforce 64-bit precision globally for numerical stability and determinism
import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
jax.config.update("jax_enable_x64", True)
