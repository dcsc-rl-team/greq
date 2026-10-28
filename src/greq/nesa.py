"""Matrix-free, per-transition NeSA with full-vector momentum."""

from __future__ import annotations

from math import isfinite
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Transition(Protocol):
    """Continuing transition consumed by :class:`NeSA`."""

    obs: jax.Array
    act: jax.Array
    rew: jax.Array
    nobs: jax.Array


@dataclass
class NeSA:
    """Apply the tabular Q-learning NeSA recurrence without a matrix estimate.

    This is the Q-learning formulation in Devraj, Busic, and Meyn,
    *Optimal Matrix Momentum Stochastic Approximation and Applications
    to Q-learning*, arXiv:1809.06277v2, Section 4 (Q-learning application).
    Costs/minima are converted to rewards/maxima. With the current greedy
    successor j and sampled pair i, A = e_i (gamma e_j - e_i)^T and
    v_next = (I + zeta A) v + zeta alpha TD e_i; Q_next = Q + v_next.
    The identity term carries ALL coordinates of v, including unsampled
    coordinates. Only the Jacobian/TD correction is sparse.

    Unlike the general nonlinear function-difference form in Equation (5),
    the paper's Q-learning formulation uses the current greedy action for
    its sampled Jacobian. The two forms need not agree at a greedy switch.
    There is no coverage barrier, frozen round table, terminal masking,
    matrix inverse, projection, or clipping. All transitions are continuing.

    The sample-clock schedule is alpha = (1 + step / k0)**(-eta), with
    step counting completed transitions. Defaults eta = k0 = zeta = 1
    give alpha = 1/(step+1) and the paper's unit momentum scale. The
    power schedules are finite-budget choices; the linear analysis does not
    establish a finite-budget guarantee for nonlinear Markovian Q-learning.

    Source: https://arxiv.org/abs/1809.06277

    Attributes:
        n_states: Number of states, S.
        n_actions: Number of actions, A.
        gamma: Discount factor in (0, 1).
        eta: Positive finite sample-clock exponent.
        k0: Positive finite sample-clock scale.
        zeta: Positive finite multiplier of the sampled correction.

    Public dataclasses:
        State: Q-table, full momentum, and completed transition count.

    Public methods:
        init: Initialize from an explicitly supplied Q-table and zero momentum.
        update: Consume one continuing transition.
        update_with: Apply host-validated mapped hyperparameters.
    """

    n_states: int
    n_actions: int
    gamma: float
    eta: float = 1.0
    k0: float = 1.0
    zeta: float = 1.0

    @dataclass
    class State:
        """Array-only state that persists across observations and checkpoints.

        Attributes:
            q: Action values, (A, S).
            velocity: Entire previous parameter increment, (A, S).
            step: Completed transitions, scalar.
        """

        q: jax.Array
        velocity: jax.Array
        step: jax.Array

    def __post_init__(self) -> None:
        """Validate dimensions, discount, and scalar hyperparameters."""
        if self.n_states < 1 or self.n_actions < 1:
            raise ValueError("state and action counts must be positive")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must lie in (0, 1)")
        for name in ("eta", "k0", "zeta"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def init(self, q: jax.Array) -> NeSA.State:
        """Initialize Q_{-1} = Q_0, hence zero parameter increment."""
        chex.assert_shape(q, (self.n_actions, self.n_states))
        if not jnp.issubdtype(q.dtype, jnp.floating):
            raise ValueError("q must have floating-point dtype")
        return self.State(
            q=q,
            velocity=jnp.zeros_like(q),
            step=jnp.zeros((), dtype=jnp.int32),
        )

    def update(self, transition: Transition, state: NeSA.State) -> NeSA.State:
        """Advance every momentum coordinate from one sampled transition."""
        return self.update_with(
            transition, state, self.gamma, self.eta, self.k0, self.zeta
        )

    def update_with(
        self,
        transition: Transition,
        state: NeSA.State,
        gamma: float | jax.Array,
        eta: float | jax.Array,
        k0: float | jax.Array,
        zeta: float | jax.Array | None = None,
    ) -> NeSA.State:
        """Use mapped parameters; an omitted zeta retains the component setting."""
        chex.assert_shape(state.q, (self.n_actions, self.n_states))
        chex.assert_equal_shape([state.q, state.velocity])
        zeta = self.zeta if zeta is None else zeta
        chex.assert_rank(
            [
                transition.obs,
                transition.act,
                transition.rew,
                transition.nobs,
                state.step,
                gamma,
                eta,
                k0,
                zeta,
            ],
            0,
        )
        gamma, eta, k0, zeta = (
            jnp.asarray(value, dtype=state.q.dtype) for value in (gamma, eta, k0, zeta)
        )
        a, s = transition.act, transition.obs
        greedy = jnp.argmax(state.q[:, transition.nobs])
        alpha = (1 + state.step.astype(state.q.dtype) / k0) ** (-eta)
        td = (
            transition.rew.astype(state.q.dtype)
            + gamma * state.q[greedy, transition.nobs]
            - state.q[a, s]
        )
        correction = zeta * (
            gamma * state.velocity[greedy, transition.nobs]
            - state.velocity[a, s]
            + alpha * td
        )
        velocity = state.velocity.at[a, s].add(correction)
        return self.State(q=state.q + velocity, velocity=velocity, step=state.step + 1)
