"""Algorithm 1 of the GReQ paper for continuing finite MDPs."""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Transition(Protocol):
    """Continuing transition consumed by :class:`Greq`."""

    obs: jax.Array
    act: jax.Array
    rew: jax.Array
    nobs: jax.Array


@dataclass
class Greq:
    """Precondition sampled TD errors with a learned greedy resolvent.

    The Q-update uses the old gain, before incorporating the current sample.
    Ties select the lowest action index. Flattening uses ``i = a * S + s``.
    The update is quadratic in ``D = A * S`` and does not form an inverse.
    This component implements a continuing MDP, without terminal masking,
    projection, denominator clipping, or automatic inverse recomputation.

    Required protocols::

        transition.obs, transition.act, transition.rew, transition.nobs

    Attributes:
        n_states: Number of states, ``S``.
        n_actions: Number of actions, ``A``.
        gamma: Discount factor in ``(0, 1)``.
        eta: Local Q-step exponent in ``(rho, 1]``.
        rho: Local model-step exponent in ``(1/2, eta)``.

    Public dataclasses:
        State: Q-table, model, gain, and local visit counts.

    Public methods:
        init: Initialize from an explicitly supplied Q-table.
        update: Consume one transition and return the new state.
    """

    n_states: int
    n_actions: int
    gamma: float
    eta: float = 1.0
    rho: float = 0.7

    @dataclass
    class State:
        """Dynamic learning state.

        Attributes:
            q: Action values, ``(A, S)``.
            model: Row-substochastic greedy transition estimate, ``(D, D)``.
            gain: Resolvent of ``model``, ``(D, D)``.
            visits: Local state-action visit counts, ``(A, S)``.
        """

        q: jax.Array
        model: jax.Array
        gain: jax.Array
        visits: jax.Array

    def __post_init__(self) -> None:
        """Validate the dimensions, discount, and separated timescales."""
        if self.n_states < 1 or self.n_actions < 1:
            raise ValueError("state and action counts must be positive")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")
        if not 0.5 < self.rho < self.eta <= 1.0:
            raise ValueError("exponents must satisfy 1/2 < rho < eta <= 1")

    def init(self, q: jax.Array) -> Greq.State:
        """Initialize the zero model, identity gain, and zero visit counts."""
        chex.assert_shape(q, (self.n_actions, self.n_states))
        if not jnp.issubdtype(q.dtype, jnp.floating):
            raise ValueError("q must have floating-point dtype")
        size = self.n_actions * self.n_states
        return self.State(
            q=q,
            model=jnp.zeros((size, size), dtype=q.dtype),
            gain=jnp.eye(size, dtype=q.dtype),
            visits=jnp.zeros(q.shape, dtype=jnp.int32),
        )

    def update(self, transition: Transition, state: Greq.State) -> Greq.State:
        """Apply one pre-update-gain Q step and one Sherman--Morrison step.

        Args:
            transition: Scalar state/action indices, reward, and successor index.
            state: Current learning state, with finite floating-point arrays.

        Returns:
            Updated learning state using the sampled pair's incremented count.
        """
        size = self.n_actions * self.n_states
        chex.assert_shape(state.q, (self.n_actions, self.n_states))
        chex.assert_equal_shape([state.q, state.visits])
        chex.assert_shape([state.model, state.gain], (size, size))
        chex.assert_rank(
            [transition.obs, transition.act, transition.rew, transition.nobs], 0
        )
        i = transition.act * self.n_states + transition.obs
        greedy = jnp.argmax(state.q[:, transition.nobs])
        j = greedy * self.n_states + transition.nobs
        visits = state.visits.at[transition.act, transition.obs].add(1)
        count = visits[transition.act, transition.obs].astype(state.q.dtype)
        alpha = count ** (-self.eta)
        beta = count ** (-self.rho)
        delta = (
            transition.rew.astype(state.q.dtype)
            + self.gamma * state.q[greedy, transition.nobs]
            - state.q[transition.act, transition.obs]
        )
        column = state.gain[:, i]
        q = state.q + (alpha * delta * column).reshape(state.q.shape)
        row_delta = -state.model[i]
        row_delta = row_delta.at[j].add(1)
        right = row_delta @ state.gain
        scale = self.gamma * beta
        denominator = 1 - scale * right[i]
        gain = state.gain + (scale / denominator) * jnp.outer(column, right)
        model = state.model.at[i].add(beta * row_delta)
        return self.State(q=q, model=model, gain=gain, visits=visits)
