"""Asynchronous Q-learning with the paper's global-clock schedule."""

from __future__ import annotations

from math import isfinite
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Transition(Protocol):
    """Continuing transition consumed by :class:`QLearning`."""

    obs: jax.Array
    act: jax.Array
    rew: jax.Array
    nobs: jax.Array


@dataclass
class QLearning:
    """Update one Q-table entry per sample in a continuing MDP.

    The global-clock schedule is ``alpha_k = (1 + k / k0)**(-eta)``.
    The first sample has ``k = 0``. Exponents at or below one half are
    permitted for finite-budget experiments, not as a convergence guarantee.

    Required protocols::

        transition.obs, transition.act, transition.rew, transition.nobs

    Attributes:
        n_states: Number of states, ``S``.
        n_actions: Number of actions, ``A``.
        gamma: Discount factor in ``(0, 1)``.
        eta: Positive learning-rate decay exponent.
        k0: Positive learning-rate scale.

    Public dataclasses:
        State: Q-table and completed update count.

    Public methods:
        init: Initialize from an explicitly supplied Q-table.
        update: Update the sampled state-action entry.
    """

    n_states: int
    n_actions: int
    gamma: float
    eta: float = 0.5
    k0: float = 10.0

    @dataclass
    class State:
        """Dynamic learning state.

        Attributes:
            q: Action values, ``(A, S)``.
            step: Number of completed updates, scalar.
        """

        q: jax.Array
        step: jax.Array

    def __post_init__(self) -> None:
        """Validate dimensions and the global-clock schedule."""
        if self.n_states < 1 or self.n_actions < 1:
            raise ValueError("state and action counts must be positive")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")
        if not isfinite(self.eta) or self.eta <= 0:
            raise ValueError("eta must be finite and positive")
        if not isfinite(self.k0) or self.k0 <= 0:
            raise ValueError("k0 must be finite and positive")

    def init(self, q: jax.Array) -> QLearning.State:
        """Initialize the global clock without changing the supplied Q-table."""
        chex.assert_shape(q, (self.n_actions, self.n_states))
        if not jnp.issubdtype(q.dtype, jnp.floating):
            raise ValueError("q must have floating-point dtype")
        return self.State(q=q, step=jnp.zeros((), dtype=jnp.int32))

    def update(self, transition: Transition, state: QLearning.State) -> QLearning.State:
        """Consume a scalar continuing transition and advance the global clock."""
        chex.assert_shape(state.q, (self.n_actions, self.n_states))
        chex.assert_rank(
            [
                transition.obs,
                transition.act,
                transition.rew,
                transition.nobs,
                state.step,
            ],
            0,
        )
        alpha = (1 + state.step.astype(state.q.dtype) / self.k0) ** (-self.eta)
        delta = (
            transition.rew.astype(state.q.dtype)
            + self.gamma * jnp.max(state.q[:, transition.nobs])
            - state.q[transition.act, transition.obs]
        )
        q = state.q.at[transition.act, transition.obs].add(alpha * delta)
        return self.State(q=q, step=state.step + 1)
