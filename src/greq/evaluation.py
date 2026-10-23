"""Exact continuing-MDP diagnostics using jaxdp Bellman and policy operators."""

from __future__ import annotations

from math import isfinite

import chex
import jax
import jax.numpy as jnp
from chex import dataclass
from jaxdp import bellman_optimality_operator, policy_evaluation
from jaxdp.mdp import Mdp

_bellman_q = jax.jit(bellman_optimality_operator.q)
_policy_v = jax.jit(policy_evaluation.v)


def greedy_policy(q: jax.Array) -> jax.Array:
    """Select the lowest maximizing action for each state of an ``(A, S)`` table."""
    chex.assert_rank(q, 2)
    return jax.nn.one_hot(jnp.argmax(q, axis=0), q.shape[0], axis=0, dtype=q.dtype)


def optimal_q(
    mdp: Mdp, gamma: float, value_tol: float = 1e-8, max_iters: int = 1_000
) -> jax.Array:
    """Solve by policy iteration and certify the sup-norm value-error bound.

    The host-side solver returns only when ``||Tq-q||_inf / (1-gamma)`` is
    at most ``value_tol``. Insufficient precision or iterations raise an
    certification error.
    """
    if not 0 < gamma < 1:
        raise ValueError("gamma must lie in (0, 1)")
    if not isfinite(value_tol) or value_tol <= 0 or max_iters < 1:
        raise ValueError("value_tol and max_iters must be positive and finite")
    if bool(jnp.any(mdp.terminal)):
        raise ValueError("the reference solver requires a continuing MDP")

    @jax.jit
    def improve(q: jax.Array) -> tuple[jax.Array, jax.Array]:
        next_q = policy_evaluation.q(mdp, greedy_policy(q), gamma)
        residual = jnp.max(
            jnp.abs(bellman_optimality_operator.q(mdp, next_q, gamma) - next_q)
        )
        return next_q, residual

    q = jnp.zeros((mdp.action_size, mdp.state_size), dtype=mdp.reward.dtype)
    for _ in range(max_iters):
        q, residual = improve(q)
        if float(residual) / (1 - gamma) <= value_tol:
            return q
    raise ArithmeticError("policy iteration did not certify the reference tolerance")


@dataclass
class Evaluation:
    """Read exact errors and greedy-policy return from a continuing MDP.

    Policy return is ``d0 @ V_pi`` for the selected greedy policy ``pi``
    and initial state distribution ``d0``.

    Attributes:
        gamma: Discount factor in ``(0, 1)``.

    Public dataclasses:
        State: MDP and certified optimal reference table.
        Metrics: Bellman error, value error, return, and policy gap.

    Public methods:
        init: Combine the supplied MDP and reference Q-table.
        evaluate: Compute diagnostics while preserving learning and RNG state.
    """

    gamma: float

    @dataclass
    class State:
        """Dynamic evaluation inputs.

        Attributes:
            mdp: Array-only jaxdp MDP.
            reference: Certified optimal Q-values, ``(A, S)``.
        """

        mdp: Mdp
        reference: jax.Array

    @dataclass
    class Metrics:
        """Exact model-based diagnostics.

        Attributes:
            bellman_linf: Sup norm of the Bellman optimality residual.
            value_linf: Sup norm of error against the reference Q-table.
            policy_return: Return of the selected greedy policy from ``d0``.
            policy_gap: Optimal reference return minus selected-policy return.
        """

        bellman_linf: jax.Array
        value_linf: jax.Array
        policy_return: jax.Array
        policy_gap: jax.Array

    def __post_init__(self) -> None:
        """Validate the discount factor."""
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must lie in (0, 1)")

    def init(self, mdp: Mdp, reference: jax.Array) -> Evaluation.State:
        """Combine independently initialized MDP and reference values."""
        chex.assert_shape(reference, (mdp.action_size, mdp.state_size))
        return self.State(mdp=mdp, reference=reference)

    def evaluate(self, q: jax.Array, state: Evaluation.State) -> Evaluation.Metrics:
        """Compute metrics from the Q-table, MDP, and reference values."""
        return evaluate(q, state, self.gamma)


def evaluate(
    q: jax.Array, state: Evaluation.State, gamma: float | jax.Array
) -> Evaluation.Metrics:
    """Evaluate with a host-validated scalar discount, including mapped discounts."""
    chex.assert_equal_shape([q, state.reference])
    chex.assert_rank(gamma, 0)
    target = _bellman_q(state.mdp, q, gamma)
    value = _policy_v(state.mdp, greedy_policy(q), gamma)
    policy_return = state.mdp.initial @ value
    optimum = state.mdp.initial @ jnp.max(state.reference, axis=0)
    return Evaluation.Metrics(
        bellman_linf=jnp.max(jnp.abs(target - q)),
        value_linf=jnp.max(jnp.abs(q - state.reference)),
        policy_return=policy_return,
        policy_gap=optimum - policy_return,
    )
