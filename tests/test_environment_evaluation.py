"""Environment conventions and independent finite-MDP evaluation checks."""

from __future__ import annotations

import itertools
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxdp.mdp import Mdp
from jaxtor.env.tabular import Mdp as TabularMdp

from greq.environment import Garnet, require_irreducible
from greq.evaluation import Evaluation, optimal_q
from scripts.run import Environment


def test_garnet_shapes_rewards_and_seed_repeatability():
    """State-action rewards are constant across successor states."""
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


def test_upstream_padded_graph_matches_independent_transition_and_reward_table():
    config = Environment(environment="graph")
    assert (config.n_states, config.n_actions, config.branching) == (6, 6, 0)
    assert "padded" in config.title
    env = config.make_environment()
    state = env.init(jax.random.key(0))
    mdp = state.mdp
    edges = ((0, 4), (1, 3, 5), (2, 3), (1, 2, 3, 4), (0, 3, 4, 5), (1, 4, 5))
    transition = np.zeros((6, 6, 6))
    reward = np.zeros((6, 6, 6))
    for s, actions in enumerate(edges):
        for a in range(6):
            target = a if a in actions else s
            for next_s in actions:
                transition[a, next_s, s] = (
                    0.8 if next_s == target else 0.2 / (len(actions) - 1)
                )
            value = 0.0 if target == s else -0.05
            if target == 5:
                value = 1.0
            if (target, s) == (4, 3):
                value = -1.0
            reward[a, s, :] = value
    np.testing.assert_allclose(mdp.transition, transition, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(mdp.reward, reward)
    np.testing.assert_array_equal(mdp.initial, np.ones(6) / 6)
    np.testing.assert_array_equal(mdp.terminal, np.zeros(6))
    for a, b in zip(
        jax.tree.leaves(mdp),
        jax.tree.leaves(env.init(jax.random.key(99)).mdp),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    require_irreducible(mdp)
    _, state = env.reset(jax.random.key(3), state)
    step, _ = env.step(
        jax.random.key(4), jnp.array(0), replace(state, step=jnp.array(1000))
    )
    assert not bool(step.trun)
    assert not bool(step.term)


@pytest.mark.parametrize("gamma", [0.95, 0.999])
def test_graph_reference_against_numpy_value_iteration(gamma):
    model = (
        Environment(environment="graph").make_environment().init(jax.random.key(0)).mdp
    )
    transition, reward = np.asarray(model.transition), np.asarray(model.reward)
    q = np.zeros((6, 6))
    for _ in range(50_000):
        following = np.einsum(
            "ans,asn->as", transition, reward + gamma * q.max(axis=0)[None, None, :]
        )
        if np.max(np.abs(following - q)) <= 1e-11:
            q = following
            break
        q = following
    else:
        pytest.fail("independent value iteration did not converge")
    mdp = Mdp(
        transition=model.transition,
        reward=model.reward,
        initial=model.initial,
        terminal=model.terminal,
    )
    actual = optimal_q(mdp, gamma)
    np.testing.assert_allclose(actual, q, rtol=0, atol=3e-8)


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
    """Certify the reference against exhaustive policy enumeration."""
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


def test_policy_return_uses_the_selected_policy_value():
    """Evaluate the policy selected by a Q-table with suboptimal greedy actions."""
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


def test_solver_requires_reference_certification():
    with pytest.raises(ArithmeticError, match="certify"):
        optimal_q(_mdp(), 0.9, value_tol=1e-30, max_iters=1)
