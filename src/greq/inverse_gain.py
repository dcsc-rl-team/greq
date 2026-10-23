"""Explicit inversion policy for dense matrix-gain learners."""

from __future__ import annotations

from math import isfinite

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


@dataclass
class InverseGain:
    """Apply a dense inverse with an optional spectral-norm bound.

    For ``J = U diag(s) V.T``, the bounded operator is
    ``V diag(1 / max(s, 1 / max_norm)) U.T``. This leaves the inverse
    unchanged when all singular values exceed the floor. Only the operator
    used for the Q-update is projected; the Jacobian estimate is untouched.
    This spectral projection implements the source paper's bounded-gain
    requirement. The choice of projection is specific to this implementation.
    ``max_norm=None`` selects an unregularized solve.

    Attributes:
        max_norm: Positive finite bound on the gain's spectral norm, or ``None``.

    Public methods:
        apply: Apply the inverse policy to one right-hand-side vector.
    """

    max_norm: float | None = 1e6

    def __post_init__(self) -> None:
        """Reject invalid gain bounds."""
        if self.max_norm is not None and (
            not isfinite(self.max_norm) or self.max_norm <= 0
        ):
            raise ValueError("max_norm must be finite and positive, or None")

    def apply(self, jacobian: jax.Array, vector: jax.Array) -> jax.Array:
        """Apply an ``(D, D)`` inverse operator to a ``(D,)`` vector."""
        chex.assert_rank(jacobian, 2)
        chex.assert_shape(jacobian, (vector.size, vector.size))
        chex.assert_shape(vector, (jacobian.shape[0],))
        if self.max_norm is None:
            return jnp.linalg.solve(jacobian, vector)
        u, singular_values, vt = jnp.linalg.svd(jacobian, full_matrices=False)
        singular_values = jnp.maximum(singular_values, 1 / self.max_norm)
        return vt.T @ ((u.T @ vector) / singular_values)
