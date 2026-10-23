"""Configure float64 for numerical correctness checks."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
