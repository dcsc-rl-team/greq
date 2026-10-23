"""Independent Zap recurrences, projection, and batch-clock regressions."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chex import dataclass

from greq.batched_zap import BatchedZap
from greq.inverse_gain import InverseGain
from greq.zap import Zap


@dataclass
class Sample:
    """Continuing tabular transition."""

    obs: jax.Array
    act: jax.Array
    rew: jax.Array
    nobs: jax.Array


def sample(s=0, a=0, r=1.0, next_s=0):
    return Sample(
        obs=jnp.array(s), act=jnp.array(a), rew=jnp.array(r), nobs=jnp.array(next_s)
    )


def inverse(matrix, vector, bound):
    """Independent NumPy reference for the explicit gain policy."""
    if bound is None:
        return np.linalg.solve(matrix, vector)
    u, singular_values, vt = np.linalg.svd(matrix)
    projected = (u * np.maximum(singular_values, 1 / bound)) @ vt
    return np.linalg.solve(projected, vector)


@pytest.mark.parametrize("bound", [None, 10.0, 1e6])
@pytest.mark.parametrize("gamma", [0.9, 0.999])
def test_zap_against_numpy_recurrence(bound, gamma):
    rng = np.random.default_rng(91)
    q = rng.normal(size=(2, 3))
    matrix = np.eye(6)
    agent = Zap(
        n_states=3,
        n_actions=2,
        gamma=gamma,
        gain=InverseGain(max_norm=bound),
        eta=0.9,
        rho=0.6,
    )
    state = agent.init(jnp.asarray(q))
    update = jax.jit(agent.update)
    for k in range(120):
        s, next_s = rng.integers(3, size=2)
        a = rng.integers(2)
        reward = rng.normal()
        greedy = q[:, next_s].argmax()
        i, j = a * 3 + s, greedy * 3 + next_s
        td = reward + gamma * q[greedy, next_s] - q[a, s]
        sampled = np.zeros((6, 6))
        sampled[i, i] += 1
        sampled[i, j] -= gamma
        beta = (k + 2) ** -0.6
        matrix = (1 - beta) * matrix + beta * sampled
        vector = np.zeros(6)
        vector[i] = td
        q = q + (k + 2) ** -0.9 * inverse(matrix, vector, bound).reshape(2, 3)
        state = update(sample(s, a, reward, next_s), state)
        np.testing.assert_allclose(state.q, q, rtol=2e-8, atol=2e-7)
        np.testing.assert_allclose(state.jacobian, matrix, rtol=1e-13, atol=1e-13)
        assert int(state.step) == k + 1


def test_updated_gain_and_all_row_decay():
    agent = Zap(n_states=2, n_actions=1, gamma=0.9, gain=InverseGain(max_norm=None))
    initial = agent.init(jnp.zeros((1, 2)))
    state = agent.update(sample(r=2), initial)
    beta = 2**-0.7
    np.testing.assert_allclose(state.jacobian, np.diag([1 - 0.9 * beta, 1 - beta]))
    np.testing.assert_allclose(state.q, [[1 / (1 - 0.9 * beta), 0]])
    np.testing.assert_array_equal(initial.jacobian, np.eye(2))


@pytest.mark.parametrize("bound", [None, 0.1, 5.0, 1e6])
@pytest.mark.parametrize("batch_size", [1, 4, 9])
def test_batched_zap_against_numpy_oracle(bound, batch_size):
    rng = np.random.default_rng(8)
    q = rng.normal(size=(2, 2))
    matrix = np.eye(4) * 0.3
    agent = BatchedZap(
        n_states=2,
        n_actions=2,
        gamma=0.99,
        gain=InverseGain(max_norm=bound),
        batch_size=batch_size,
        eta=0.9,
        rho=0.6,
        initial_scale=0.3,
    )
    state = agent.init(jnp.asarray(q))
    update = jax.jit(agent.update)
    td_sum = np.zeros(4)
    matrix_sum = np.zeros((4, 4))
    survival = 1.0
    batches = 0
    for k in range(4 * batch_size + batch_size // 2):
        s, a, next_s = rng.integers(2, size=3)
        reward = rng.normal()
        greedy = q[:, next_s].argmax()
        i, j = a * 2 + s, greedy * 2 + next_s
        td_sum[i] += reward + 0.99 * q[greedy, next_s] - q[a, s]
        matrix_sum[i, i] += 1
        matrix_sum[i, j] -= 0.99
        survival *= 1 - (k + 2) ** -0.6
        if (k + 1) % batch_size == 0:
            weight = 1 - survival
            matrix = survival * matrix + weight * matrix_sum / batch_size
            q = q + (batches + 2) ** -0.9 * inverse(
                matrix, td_sum / batch_size, bound
            ).reshape(q.shape)
            batches += 1
            td_sum = np.zeros(4)
            matrix_sum = np.zeros((4, 4))
            survival = 1.0
        state = update(sample(s, a, reward, next_s), state)
        np.testing.assert_allclose(state.q, q, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(state.jacobian, matrix, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(state.td_sum, td_sum, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(state.jacobian_sum, matrix_sum, rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.exp(state.log_survival), survival, rtol=1e-13)
        assert int(state.batches) == batches
        assert int(state.batch_step) == (k + 1) % batch_size
        assert int(state.step) == k + 1


def test_batch_holds_q_and_accumulates_every_sample_until_boundary():
    agent = BatchedZap(
        n_states=1, n_actions=2, gamma=0.9, gain=InverseGain(), batch_size=3
    )
    initial = agent.init(jnp.zeros((2, 1)))
    state = agent.update(sample(a=1, r=100), initial)
    state = agent.update(sample(a=1, r=100), state)
    np.testing.assert_array_equal(state.q, initial.q)
    np.testing.assert_array_equal(state.jacobian, initial.jacobian)
    np.testing.assert_allclose(state.td_sum, [0, 200])
    np.testing.assert_allclose(state.jacobian_sum, [[0, 0], [-1.8, 2]])
    state = agent.update(sample(a=1, r=100), state)
    assert not np.array_equal(state.q, initial.q)
    np.testing.assert_array_equal(state.td_sum, np.zeros(2))
    np.testing.assert_array_equal(state.jacobian_sum, np.zeros((2, 2)))


@pytest.mark.parametrize("bound", [None, 5.0, 1e6])
def test_batch_one_matches_full_zap(bound):
    full = Zap(n_states=2, n_actions=2, gamma=0.9, gain=InverseGain(max_norm=bound))
    batch = BatchedZap(
        n_states=2,
        n_actions=2,
        gamma=0.9,
        gain=InverseGain(max_norm=bound),
        batch_size=1,
    )
    a = full.init(jnp.zeros((2, 2)))
    b = batch.init(jnp.zeros((2, 2)))
    for k in range(16):
        transition = sample(s=k % 2, a=(k // 2) % 2, r=k / 16, next_s=1 - k % 2)
        a = full.update(transition, a)
        b = batch.update(transition, b)
        np.testing.assert_allclose(a.q, b.q, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(a.jacobian, b.jacobian, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_transformations_and_dtype(batched, dtype):
    kwargs = dict(n_states=2, n_actions=2, gamma=0.9, gain=InverseGain())
    agent = BatchedZap(**kwargs, batch_size=3) if batched else Zap(**kwargs)
    initial = agent.init(jnp.zeros((2, 2), dtype=dtype))
    transitions = Sample(
        obs=jnp.array([0, 1, 0, 1]),
        act=jnp.array([1, 0, 0, 1]),
        rew=jnp.array([1.0, 2.0, 3.0, 4.0]),
        nobs=jnp.array([1, 0, 1, 0]),
    )

    def train(state):
        return jax.lax.scan(
            lambda s, t: (agent.update(t, s), None), state, transitions
        )[0]

    eager = train(initial)
    compiled = jax.jit(train)(initial)
    vectorized = jax.jit(jax.vmap(train))(
        jax.vmap(agent.init)(jnp.zeros((2, 2, 2), dtype=dtype))
    )
    for a, b, c in zip(
        jax.tree.leaves(eager),
        jax.tree.leaves(compiled),
        jax.tree.leaves(vectorized),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(c, np.broadcast_to(a, c.shape), rtol=2e-5, atol=2e-5)
    assert compiled.q.dtype == dtype
    assert compiled.jacobian.dtype == dtype
    if batched:
        assert compiled.td_sum.dtype == dtype
        assert compiled.jacobian_sum.dtype == dtype


def test_gain_projection_bounds_inverse_without_changing_matrix():
    matrix = jnp.array([[1e-10, 0.0], [0.0, 2.0]])
    gain = InverseGain(max_norm=7)
    operator = jax.vmap(lambda v: gain.apply(matrix, v), in_axes=1, out_axes=1)(
        jnp.eye(2)
    )
    np.testing.assert_allclose(operator, np.diag([7, 0.5]))
    assert np.linalg.norm(operator, ord=2) <= 7 + 1e-12
    np.testing.assert_array_equal(matrix, [[1e-10, 0], [0, 2]])
    singular = gain.apply(jnp.zeros((2, 2)), jnp.ones(2))
    assert np.isfinite(singular).all()


@pytest.mark.parametrize("bound", [0, -1, float("inf"), float("nan")])
def test_invalid_gain_bound(bound):
    with pytest.raises(ValueError, match="max_norm"):
        InverseGain(max_norm=bound)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "option,value",
    [
        ("gamma", 1),
        ("rho", 0.5),
        ("eta", 0.6),
        ("initial_scale", 0),
        ("initial_scale", float("nan")),
    ],
)
def test_invalid_settings(batched, option, value):
    kwargs = dict(n_states=2, n_actions=2, gamma=0.9, gain=InverseGain())
    kwargs[option] = value
    with pytest.raises(ValueError):
        BatchedZap(**kwargs, batch_size=3) if batched else Zap(**kwargs)


@pytest.mark.parametrize("period", [0, -1, 1.5, 2**31])
def test_invalid_batch_size(period):
    with pytest.raises(ValueError, match="batch_size"):
        BatchedZap(
            n_states=2, n_actions=2, gamma=0.9, gain=InverseGain(), batch_size=period
        )


@pytest.mark.parametrize("batched", [False, True])
def test_invalid_shapes_and_initial_dtype(batched):
    kwargs = dict(n_states=2, n_actions=2, gamma=0.9, gain=InverseGain())
    agent = BatchedZap(**kwargs, batch_size=3) if batched else Zap(**kwargs)
    with pytest.raises(ValueError, match="floating"):
        agent.init(jnp.zeros((2, 2), dtype=jnp.int32))
    with pytest.raises(AssertionError):
        agent.init(jnp.zeros(4))
    state = agent.init(jnp.zeros((2, 2)))
    with pytest.raises(AssertionError):
        agent.update(replace(sample(), rew=jnp.ones(3)), state)
