"""Batched tabular Zap, equations (74)--(75) of arXiv:1707.03770v2."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Transition(Protocol):
    """Continuing transition consumed by :class:`BatchedZap`."""

    obs: jax.Array
    act: jax.Array
    rew: jax.Array
    nobs: jax.Array


class Gain(Protocol):
    """Inverse application consumed by :class:`BatchedZap`."""

    def apply(self, jacobian: jax.Array, vector: jax.Array, /) -> jax.Array: ...


@dataclass
class BatchedZap:
    """Hold Q fixed for a batch and apply one updated matrix gain at its end.

    For batch ``b``, use ``alpha_b = (b+2)**(-eta)`` and
    ``beta_bar = 1 - product_k(1 - (k+2)**(-rho))`` over its sample indices.
    The TD vector and sampled Jacobian are arithmetic batch means.
    Incomplete batches persist across evaluation and checkpointing until
    the next batch boundary. A batch size of one recovers
    :class:`~greq.zap.Zap` up to floating-point rounding.

    Required protocols::

        transition.obs, transition.act, transition.rew, transition.nobs
        gain.apply(jacobian, mean_td_vector) -> direction

    Attributes:
        n_states: State count, ``S``.
        n_actions: Action count, ``A``; matrix index is ``a * S + s``.
        gamma: Continuing-task discount in ``(0, 1)``.
        gain: Explicit inverse or bounded-inverse operator.
        batch_size: Samples per matrix and Q-update, ``N``.
        eta: Batch-clock Q exponent in ``(rho, 1]``.
        rho: Sample-clock matrix exponent in ``(1/2, eta)``.
        initial_scale: Positive initial Jacobian scale.

    Public dataclasses:
        State: Learning state, clocks, and incomplete-batch accumulators.

    Public methods:
        init: Initialize a Q-table and empty batch.
        update: Accumulate a sample and update Q only at a batch boundary.
        update_with: Accumulate with scalar array parameters and shared clocks.
    """

    n_states: int
    n_actions: int
    gamma: float
    gain: Gain
    batch_size: int
    eta: float = 1.0
    rho: float = 0.7
    initial_scale: float = 1.0

    @dataclass
    class State:
        """Dynamic batched Zap state.

        Attributes:
            q: Q-table held fixed inside each batch, ``(A, S)``.
            jacobian: Unprojected Jacobian from the last boundary, ``(D, D)``.
            step: Total number of consumed samples.
            batches: Number of completed batches.
            batch_step: Number of samples in the incomplete batch.
            td_sum: Sum of TD-feature vectors in the incomplete batch, ``(D,)``.
            jacobian_sum: Sum of sampled positive-sign Jacobians, ``(D, D)``.
            log_survival: Sum of ``log(1-beta_k)`` in the incomplete batch.
        """

        q: jax.Array
        jacobian: jax.Array
        step: jax.Array
        batches: jax.Array
        batch_step: jax.Array
        td_sum: jax.Array
        jacobian_sum: jax.Array
        log_survival: jax.Array

    def __post_init__(self) -> None:
        """Validate dimensions, clocks, batch size, and initialization."""
        if self.n_states < 1 or self.n_actions < 1:
            raise ValueError("state and action counts must be positive")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must lie in (0, 1)")
        if not 0.5 < self.rho < self.eta <= 1:
            raise ValueError("exponents must satisfy 1/2 < rho < eta <= 1")
        if (
            not isinstance(self.batch_size, int)
            or not 1 <= self.batch_size <= 2_147_483_647
        ):
            raise ValueError("batch_size must fit a positive int32")
        if not isfinite(self.initial_scale) or self.initial_scale <= 0:
            raise ValueError("initial_scale must be finite and positive")

    def init(self, q: jax.Array) -> BatchedZap.State:
        """Initialize a scaled identity Jacobian, zero clocks, and empty batch."""
        chex.assert_shape(q, (self.n_actions, self.n_states))
        if not jnp.issubdtype(q.dtype, jnp.floating):
            raise ValueError("q must have floating-point dtype")
        size = self.n_states * self.n_actions
        return self.State(
            q=q,
            jacobian=self.initial_scale * jnp.eye(size, dtype=q.dtype),
            step=jnp.zeros((), dtype=jnp.int32),
            batches=jnp.zeros((), dtype=jnp.int32),
            batch_step=jnp.zeros((), dtype=jnp.int32),
            td_sum=jnp.zeros((size,), dtype=q.dtype),
            jacobian_sum=jnp.zeros((size, size), dtype=q.dtype),
            log_survival=jnp.zeros((), dtype=q.dtype),
        )

    def update(
        self, transition: Transition, state: BatchedZap.State
    ) -> BatchedZap.State:
        """Accumulate at the frozen Q-table and finish only complete batches."""
        return self.update_with(transition, state, self.gamma, self.eta, self.rho)

    def update_with(
        self,
        transition: Transition,
        state: BatchedZap.State,
        gamma: float | jax.Array,
        eta: float | jax.Array,
        rho: float | jax.Array,
    ) -> BatchedZap.State:
        """Accumulate with mapped rates while keeping the batch predicate scalar."""
        size = self.n_states * self.n_actions
        chex.assert_shape(state.q, (self.n_actions, self.n_states))
        chex.assert_shape([state.jacobian, state.jacobian_sum], (size, size))
        chex.assert_shape(state.td_sum, (size,))
        chex.assert_rank(
            [
                transition.obs,
                transition.act,
                transition.rew,
                transition.nobs,
                state.step,
                state.batches,
                state.batch_step,
                state.log_survival,
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
        beta = (state.step.astype(state.q.dtype) + 2) ** (-rho)
        state = replace(
            state,
            step=state.step + 1,
            batch_step=state.batch_step + 1,
            td_sum=state.td_sum.at[i].add(delta),
            jacobian_sum=state.jacobian_sum.at[i, i].add(1).at[i, j].add(-gamma),
            log_survival=state.log_survival + jnp.log1p(-beta),
        )
        return jax.lax.cond(
            state.batch_step == self.batch_size,
            lambda value: self._finish(value, eta),
            lambda value: value,
            state,
        )

    def _finish(
        self, state: BatchedZap.State, eta: float | jax.Array
    ) -> BatchedZap.State:
        """Apply arithmetic batch means using the product-aggregated fast step."""
        beta = -jnp.expm1(state.log_survival)
        jacobian = (
            1 - beta
        ) * state.jacobian + beta * state.jacobian_sum / self.batch_size
        alpha = (state.batches.astype(state.q.dtype) + 2) ** (-eta)
        direction = self.gain.apply(jacobian, state.td_sum / self.batch_size)
        return replace(
            state,
            q=state.q + alpha * direction.reshape(state.q.shape),
            jacobian=jacobian,
            batches=state.batches + 1,
            batch_step=jnp.zeros_like(state.batch_step),
            td_sum=jnp.zeros_like(state.td_sum),
            jacobian_sum=jnp.zeros_like(state.jacobian_sum),
            log_survival=jnp.zeros_like(state.log_survival),
        )
