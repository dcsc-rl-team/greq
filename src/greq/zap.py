"""Tabular Zap Q-learning from Devraj and Meyn, arXiv:1707.03770v2."""

from __future__ import annotations

from math import isfinite
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Transition(Protocol):
    """Continuing transition consumed by :class:`Zap`."""

    obs: jax.Array
    act: jax.Array
    rew: jax.Array
    nobs: jax.Array


class Gain(Protocol):
    """Inverse application consumed by :class:`Zap`."""

    def apply(self, jacobian: jax.Array, vector: jax.Array, /) -> jax.Array: ...


@dataclass
class Zap:
    """Update a global Jacobian estimate and apply its updated inverse each sample.

    This is the reward-maximization form of Algorithm 1 with lambda zero and
    tabular features. The stored matrix is ``J = -A`` in the source's notation.
    Both clocks are global: ``alpha_k = (k+2)**(-eta)`` and
    ``beta_k = (k+2)**(-rho)``. The offset follows the GReQ experiment schedule.
    Every Jacobian row decays at every sample, and the Q-update uses the
    updated Jacobian. Ties choose the lowest action index.

    Required protocols::

        transition.obs, transition.act, transition.rew, transition.nobs
        gain.apply(jacobian, td_vector) -> direction

    Attributes:
        n_states: State count, ``S``.
        n_actions: Action count, ``A``; matrix index is ``a * S + s``.
        gamma: Continuing-task discount in ``(0, 1)``.
        gain: Explicit inverse or bounded-inverse operator.
        eta: Slow exponent in ``(rho, 1]``.
        rho: Fast exponent in ``(1/2, eta)``.
        initial_scale: Positive scale for the initial Jacobian ``J0 = scale * I``.

    Public dataclasses:
        State: Q-table, unprojected Jacobian, and global clock.

    Public methods:
        init: Initialize from a supplied Q-table.
        update: Consume one sample and apply the updated matrix gain.
        update_with: Apply the same recurrence with scalar array parameters.
    """

    n_states: int
    n_actions: int
    gamma: float
    gain: Gain
    eta: float = 1.0
    rho: float = 0.7
    initial_scale: float = 1.0

    @dataclass
    class State:
        """Dynamic Zap state.

        Attributes:
            q: Action values, ``(A, S)``.
            jacobian: Unprojected positive-sign Jacobian estimate, ``(D, D)``.
            step: Number of consumed samples, scalar.
        """

        q: jax.Array
        jacobian: jax.Array
        step: jax.Array

    def __post_init__(self) -> None:
        """Validate dimensions, timescales, and Jacobian initialization."""
        if self.n_states < 1 or self.n_actions < 1:
            raise ValueError("state and action counts must be positive")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must lie in (0, 1)")
        if not 0.5 < self.rho < self.eta <= 1:
            raise ValueError("exponents must satisfy 1/2 < rho < eta <= 1")
        if not isfinite(self.initial_scale) or self.initial_scale <= 0:
            raise ValueError("initial_scale must be finite and positive")

    def init(self, q: jax.Array) -> Zap.State:
        """Initialize the Jacobian to a scaled identity and reset the sample clock."""
        chex.assert_shape(q, (self.n_actions, self.n_states))
        if not jnp.issubdtype(q.dtype, jnp.floating):
            raise ValueError("q must have floating-point dtype")
        size = self.n_actions * self.n_states
        return self.State(
            q=q,
            jacobian=self.initial_scale * jnp.eye(size, dtype=q.dtype),
            step=jnp.zeros((), dtype=jnp.int32),
        )

    def update(self, transition: Transition, state: Zap.State) -> Zap.State:
        """Decay all Jacobian rows, incorporate one sample, and update Q."""
        return self.update_with(transition, state, self.gamma, self.eta, self.rho)

    def update_with(
        self,
        transition: Transition,
        state: Zap.State,
        gamma: float | jax.Array,
        eta: float | jax.Array,
        rho: float | jax.Array,
    ) -> Zap.State:
        """Apply the recurrence with host-validated, potentially mapped parameters."""
        size = self.n_states * self.n_actions
        chex.assert_shape(state.q, (self.n_actions, self.n_states))
        chex.assert_shape(state.jacobian, (size, size))
        chex.assert_rank(
            [
                transition.obs,
                transition.act,
                transition.rew,
                transition.nobs,
                state.step,
                gamma,
                eta,
                rho,
            ],
            0,
        )
        i = transition.act * self.n_states + transition.obs
        action = jnp.argmax(state.q[:, transition.nobs])
        j = action * self.n_states + transition.nobs
        delta = (
            transition.rew.astype(state.q.dtype)
            + gamma * state.q[action, transition.nobs]
            - state.q[transition.act, transition.obs]
        )
        clock = state.step.astype(state.q.dtype) + 2
        alpha, beta = clock ** (-eta), clock ** (-rho)
        jacobian = (1 - beta) * state.jacobian
        jacobian = jacobian.at[i, i].add(beta).at[i, j].add(-gamma * beta)
        vector = jnp.zeros((size,), dtype=state.q.dtype).at[i].set(delta)
        q = state.q + alpha * self.gain.apply(jacobian, vector).reshape(state.q.shape)
        return self.State(q=q, jacobian=jacobian, step=state.step + 1)
