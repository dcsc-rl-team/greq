"""Independent recurrence checks and JAX transformation tests."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chex import dataclass

from greq import Greq, QLearning


@dataclass
class Sample:
    """Scalar transition or a batch of scalar transitions."""

    obs: jax.Array
    act: jax.Array
    rew: jax.Array
    nobs: jax.Array


def sample(s=0, a=0, r=1.0, next_s=0) -> Sample:
    """Construct a scalar transition."""
    return Sample(
        obs=jnp.array(s), act=jnp.array(a), rew=jnp.array(r), nobs=jnp.array(next_s)
    )


@pytest.mark.parametrize("gamma", [0.1, 0.9, 0.99, 0.999])
def test_greq_matches_direct_solve_after_every_update(gamma):
    """Compare both q and gain with an independent dense NumPy oracle."""
    agent = Greq(n_states=3, n_actions=2, gamma=gamma, eta=0.9, rho=0.6)
    rng = np.random.default_rng(6)
    q = rng.normal(size=(2, 3))
    model = np.zeros((6, 6))
    counts = np.zeros((2, 3), dtype=int)
    state = agent.init(jnp.array(q))
    update = jax.jit(agent.update)
    for _ in range(200):
        s, next_s = rng.integers(3, size=2)
        a = rng.integers(2)
        reward = rng.normal()
        i = a * 3 + s
        action = q[:, next_s].argmax()
        j = action * 3 + next_s
        counts[a, s] += 1
        alpha = float(counts[a, s]) ** -agent.eta
        beta = float(counts[a, s]) ** -agent.rho
        gain = np.linalg.solve(np.eye(6) - gamma * model, np.eye(6))
        delta = reward + gamma * q[action, next_s] - q[a, s]
        q = q + (alpha * delta * gain[:, i]).reshape(q.shape)
        model[i] = (1 - beta) * model[i] + beta * np.eye(6)[j]
        state = update(sample(s, a, reward, next_s), state)
        expected_gain = np.linalg.solve(np.eye(6) - gamma * model, np.eye(6))
        np.testing.assert_allclose(state.q, q, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(state.gain, expected_gain, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(state.model, model, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(state.visits, counts)
        assert np.min(state.model) >= 0
        assert np.max(np.sum(state.model, axis=1)) <= 1 + 1e-14
        assert np.max(np.sum(np.abs(state.gain), axis=1)) <= 1 / (1 - gamma) + 1e-7


def test_first_update_uses_old_gain_and_incremented_count():
    """A one-state self-loop separates old-gain from new-gain algorithms."""
    agent = Greq(n_states=1, n_actions=1, gamma=0.9)
    before = agent.init(jnp.zeros((1, 1)))
    after = agent.update(sample(r=2), before)
    np.testing.assert_array_equal(after.q, [[2]])
    np.testing.assert_array_equal(after.model, [[1]])
    np.testing.assert_allclose(after.gain, [[10]])
    np.testing.assert_array_equal(after.visits, [[1]])
    np.testing.assert_array_equal(before.q, [[0]])
    np.testing.assert_array_equal(before.model, [[0]])


def test_ties_and_action_major_indexing():
    """The greedy continuation selects action zero before updating q."""
    agent = Greq(n_states=2, n_actions=3, gamma=0.9)
    state = agent.init(jnp.zeros((3, 2)))
    state = agent.update(sample(s=1, a=2, r=3, next_s=1), state)
    expected = np.zeros((6, 6))
    expected[5, 1] = 1
    np.testing.assert_array_equal(state.model, expected)
    expected_q = np.zeros((3, 2))
    expected_q[2, 1] = 3
    np.testing.assert_array_equal(state.q, expected_q)


def test_local_clock_ignores_unvisited_rows():
    """A new pair uses alpha=beta=1 regardless of the global sample count."""
    agent = Greq(n_states=2, n_actions=1, gamma=0.5)
    state = agent.init(jnp.zeros((1, 2)))
    for _ in range(4):
        state = agent.update(sample(), state)
    before = state
    state = agent.update(sample(s=1, next_s=1, r=3), state)
    np.testing.assert_array_equal(state.visits, [[4, 1]])
    np.testing.assert_array_equal(state.model[0], before.model[0])
    np.testing.assert_array_equal(state.model[1], [0, 1])
    assert float(state.q[0, 1]) == 3


@pytest.mark.parametrize("method", [Greq, QLearning])
def test_jit_scan_and_vmap_agree(method):
    """Nested dataclasses transform without filtering or component mutation."""
    agent = method(n_states=2, n_actions=2, gamma=0.9)
    transitions = Sample(
        obs=jnp.array([0, 1, 0, 1]),
        act=jnp.array([0, 1, 1, 0]),
        rew=jnp.array([1.0, -1.0, 0.5, 2.0]),
        nobs=jnp.array([1, 0, 1, 0]),
    )
    initial = agent.init(jnp.zeros((2, 2)))
    eager = initial
    for k in range(4):
        eager = agent.update(jax.tree.map(lambda x, k=k: x[k], transitions), eager)

    def train(state):
        return jax.lax.scan(
            lambda s, t: (agent.update(t, s), None), state, transitions
        )[0]

    compiled = jax.jit(train)(initial)
    batch = jax.vmap(agent.init)(jnp.zeros((3, 2, 2)))
    batched = jax.jit(jax.vmap(train))(batch)
    for expected, actual, vectorized in zip(
        jax.tree.leaves(eager),
        jax.tree.leaves(compiled),
        jax.tree.leaves(batched),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(
            vectorized, np.broadcast_to(expected, vectorized.shape)
        )


def test_q_learning_schedule_and_single_entry_update():
    """The first global-clock rate is one and the next uses k=1."""
    agent = QLearning(n_states=2, n_actions=2, gamma=0.9, eta=0.5, k0=10)
    state = agent.init(jnp.zeros((2, 2)))
    state = agent.update(sample(s=0, a=1, r=2, next_s=1), state)
    np.testing.assert_array_equal(state.q, [[0, 0], [2, 0]])
    state = agent.update(sample(s=1, a=0, r=1, next_s=0), state)
    np.testing.assert_allclose(state.q[0, 1], (1 + 1 / 10) ** -0.5 * (1 + 0.9 * 2))
    assert int(state.step) == 2


@pytest.mark.parametrize("gamma", [0, 1, -1, float("nan")])
@pytest.mark.parametrize("method", [Greq, QLearning])
def test_invalid_discount(method, gamma):
    with pytest.raises(ValueError, match="gamma"):
        method(n_states=2, n_actions=2, gamma=gamma)


@pytest.mark.parametrize("eta,rho", [(1, 0.5), (0.7, 0.7), (0.6, 0.9), (1.1, 0.7)])
def test_invalid_timescales(eta, rho):
    with pytest.raises(ValueError, match="exponents"):
        Greq(n_states=2, n_actions=2, gamma=0.9, eta=eta, rho=rho)


@pytest.mark.parametrize("method", [Greq, QLearning])
def test_invalid_shapes_and_dtypes(method):
    agent = method(n_states=2, n_actions=3, gamma=0.9)
    with pytest.raises(AssertionError):
        agent.init(jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="floating"):
        agent.init(jnp.zeros((3, 2), dtype=jnp.int32))
    state = agent.init(jnp.zeros((3, 2)))
    with pytest.raises(AssertionError):
        agent.update(replace(sample(), rew=jnp.ones(2)), state)


@pytest.mark.parametrize("method", [Greq, QLearning])
def test_float32_supported(method):
    agent = method(n_states=1, n_actions=1, gamma=0.9)
    state = agent.init(jnp.zeros((1, 1), dtype=jnp.float32))
    after = jax.jit(agent.update)(sample(), state)
    assert after.q.dtype == jnp.float32
