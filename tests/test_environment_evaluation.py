"""Environment conventions and independent finite-MDP evaluation checks."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxdp.mdp import Mdp
from jaxtor.env.tabular import Mdp as TabularMdp

from greq.environment import Garnet, require_irreducible
from greq.evaluation import Evaluation, optimal_q


def test_garnet_shapes_rewards_and_seed_repeatability():
    """Rewards are state-action draws, not successor-dependent draws."""
    config = Garnet(n_states=4, n_actions=2, branching=3)
    a = config.init_mdp(jax.random.key(5))
    b = config.init_mdp(jax.random.key(5))
    c = config.init_mdp(jax.random.key(6))
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a.transition, c.transition)
    assert a.transition.shape == (2, 4, 4)
    np.testing.assert_allclose(a.transition.sum(axis=1), 1)
    np.testing.assert_array_equal((a.transition > 0).sum(axis=1), 3)
    np.testing.assert_array_equal(
        a.reward, np.broadcast_to(a.reward[:, :, :1], (2, 4, 4))
    )
    np.testing.assert_array_equal(a.initial, np.full(4, 0.25))
    np.testing.assert_array_equal(a.terminal, np.zeros(4))
    require_irreducible(a)


def test_reducible_instances_fail_without_resampling():
    """Two disconnected self-loops cannot provide full behavior coverage."""
    mdp = TabularMdp(
        transition=jnp.eye(2)[None],
        reward=jnp.zeros((1, 2, 2)),
        initial=jnp.ones(2) / 2,
        terminal=jnp.zeros(2),
    )
    with pytest.raises(ValueError, match="irreducible"):
        require_irreducible(mdp)


@pytest.mark.parametrize("branching", [0, 4])
def test_invalid_garnet_branching(branching):
    with pytest.raises(ValueError, match="branching"):
        Garnet(n_states=3, branching=branching)


def _mdp() -> Mdp:
    """Return a nontrivial two-action, three-state continuing MDP."""
    transition = jnp.array(
        [
            [[0.8, 0.1, 0.0], [0.2, 0.6, 0.4], [0.0, 0.3, 0.6]],
            [[0.3, 0.7, 0.2], [0.4, 0.1, 0.2], [0.3, 0.2, 0.6]],
        ]
    )
    reward = jnp.arange(18, dtype=jnp.float64).reshape(2, 3, 3) / 17
    return Mdp(
        transition=transition,
        reward=reward,
        initial=jnp.array([0.2, 0.5, 0.3]),
        terminal=jnp.zeros(3),
    )


def _numpy_policy_value(mdp, actions, gamma):
    """Evaluate a deterministic policy using explicit row extraction."""
    transition = np.asarray(mdp.transition)
    reward = np.asarray(mdp.reward)
    p = np.stack([transition[a, :, s] for s, a in enumerate(actions)])
    r = np.array(
        [np.dot(transition[a, :, s], reward[a, s]) for s, a in enumerate(actions)]
    )
    return np.linalg.solve(np.eye(len(actions)) - gamma * p, r)


@pytest.mark.parametrize("gamma", [0.1, 0.9, 0.999])
def test_optimal_reference_against_enumerated_policies(gamma):
    """Enumerate every policy rather than trusting a second policy iterator."""
    mdp = _mdp()
    values = [
        _numpy_policy_value(mdp, actions, gamma)
        for actions in itertools.product(range(2), repeat=3)
    ]
    v_star = np.max(values, axis=0)
    expected = np.zeros((2, 3))
    for a in range(2):
        for s in range(3):
            expected[a, s] = np.dot(
                mdp.transition[a, :, s], mdp.reward[a, s] + gamma * v_star
            )
    q = optimal_q(mdp, gamma)
    np.testing.assert_allclose(q, expected, rtol=1e-11, atol=1e-9)
    evaluator = Evaluation(gamma=gamma)
    metrics = jax.jit(evaluator.evaluate)(q, evaluator.init(mdp, q))
    assert float(metrics.bellman_linf) / (1 - gamma) <= 1e-8
    assert abs(float(metrics.policy_gap)) <= 1e-8


def test_policy_return_evaluates_selected_action_not_max_of_q_pi():
    """A deliberately wrong greedy choice must not get a one-step improvement."""
    mdp = Mdp(
        transition=jnp.ones((2, 1, 1)),
        reward=jnp.array([[[0.0]], [[1.0]]]),
        initial=jnp.ones(1),
        terminal=jnp.zeros(1),
    )
    evaluator = Evaluation(gamma=0.9)
    reference = optimal_q(mdp, 0.9)
    state = evaluator.init(mdp, reference)
    metrics = evaluator.evaluate(jnp.array([[1.0], [0.0]]), state)
    assert float(metrics.policy_return) == 0
    np.testing.assert_allclose(metrics.policy_gap, 10)


def test_solver_does_not_silently_return_uncertified_reference():
    with pytest.raises(ArithmeticError, match="certify"):
        optimal_q(_mdp(), 0.9, value_tol=1e-30, max_iters=1)
