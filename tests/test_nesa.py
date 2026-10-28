"""Independent dense-matrix NeSA oracle and full-momentum regressions."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chex import dataclass

from greq import NeSA


@dataclass
class Sample:
    """Continuing tabular transition, optionally with leading axes."""

    obs: jax.Array
    act: jax.Array
    rew: jax.Array
    nobs: jax.Array


def sample(s=0, a=0, r=1.0, next_s=0):
    return Sample(
        obs=jnp.array(s), act=jnp.array(a), rew=jnp.array(r), nobs=jnp.array(next_s)
    )


@pytest.mark.parametrize("gamma", [0.1, 0.9, 0.999])
@pytest.mark.parametrize("eta,k0,zeta", [(1, 1, 1), (0.6, 10, 0.25), (0.8, 1, 0.75)])
def test_nesa_matches_primary_dense_matrix_recurrence(gamma, eta, k0, zeta):
    rng = np.random.default_rng(52)
    q = rng.normal(size=(2, 3))
    velocity = np.zeros(6)
    agent = NeSA(n_states=3, n_actions=2, gamma=gamma, eta=eta, k0=k0, zeta=zeta)
    initial = agent.init(jnp.asarray(q))
    state = initial
    update = jax.jit(agent.update)
    for step in range(300):
        s, successor = rng.integers(3, size=2)
        a = rng.integers(2)
        reward = rng.normal()
        i = a * 3 + s
        greedy = np.argmax(q[:, successor])
        j = greedy * 3 + successor
        basis, next_basis = np.eye(6)[i], np.eye(6)[j]
        jacobian = np.outer(basis, gamma * next_basis - basis)
        td = reward + gamma * q[greedy, successor] - q[a, s]
        alpha = (1 + step / k0) ** (-eta)
        velocity = (np.eye(6) + zeta * jacobian) @ velocity + zeta * alpha * td * basis
        q = q + velocity.reshape(q.shape)
        state = update(sample(s, a, reward, successor), state)
        np.testing.assert_allclose(state.q, q, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            state.velocity, velocity.reshape(q.shape), rtol=1e-10, atol=1e-10
        )
        assert int(state.step) == step + 1
    np.testing.assert_array_equal(initial.velocity, np.zeros((2, 3)))
    assert int(initial.step) == 0


def test_unsampled_coordinate_keeps_its_momentum_without_a_coverage_barrier():
    agent = NeSA(n_states=2, n_actions=1, gamma=0.5)
    initial = agent.init(jnp.zeros((1, 2)))
    first = agent.update(sample(r=1, next_s=1), initial)
    np.testing.assert_array_equal(first.q, [[1, 0]])
    second = agent.update(sample(s=1, r=0, next_s=0), first)
    np.testing.assert_array_equal(second.velocity, [[1, 0.75]])
    np.testing.assert_array_equal(second.q, [[2, 0.75]])
    np.testing.assert_array_equal(first.q, [[1, 0]])
    assert int(second.step) == 2
    # The archived coordinate-only adaptation produces [[1, .75]], not NeSA.


def test_sampled_jacobian_uses_current_greedy_action_even_when_it_changes():
    agent = NeSA(n_states=2, n_actions=2, gamma=0.9, zeta=0.5)
    q = jnp.array([[4.0, 1.0], [0.0, 2.0]])
    velocity = jnp.array([[1.0, -2.0], [3.0, 4.0]])
    state = replace(agent.init(q), velocity=velocity, step=jnp.array(2))
    assert int(jnp.argmax(q[:, 1])) == 1
    assert int(jnp.argmax((q - velocity)[:, 1])) == 0
    after = agent.update(sample(r=0.5, next_s=1), state)
    td = 0.5 + 0.9 * 2 - 4
    correction = 0.5 * (0.9 * 4 - 1 + td / 3)
    expected = np.asarray(velocity).copy()
    expected[0, 0] += correction
    np.testing.assert_allclose(after.velocity, expected, atol=1e-14)
    np.testing.assert_allclose(after.q, q + expected, atol=1e-14)


def test_first_step_and_global_schedule_do_not_wait_for_unvisited_pairs():
    agent = NeSA(n_states=2, n_actions=1, gamma=0.9, eta=0.8, k0=10, zeta=0.25)
    state = agent.init(jnp.zeros((1, 2)))
    for step in range(20):
        old_q, old_v = float(state.q[0, 0]), float(state.velocity[0, 0])
        alpha = (1 + step / 10) ** (-0.8)
        expected = (1 - 0.25 * 0.1) * old_v + 0.25 * alpha * (1 - 0.1 * old_q)
        state = agent.update(sample(), state)
        np.testing.assert_allclose(state.velocity[0, 0], expected, atol=1e-14)
        assert int(state.step) == step + 1
        assert float(state.q[0, 1]) == 0
    assert float(state.q[0, 0]) > 0.25


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jit_scan_and_vmap_map_all_hyperparameters_and_preserve_dtype(dtype):
    agent = NeSA(n_states=2, n_actions=2, gamma=0.9)
    initial = agent.init(jnp.zeros((2, 2), dtype=dtype))
    transitions = Sample(
        obs=jnp.array([0, 1, 0, 0, 1, 0]),
        act=jnp.array([0, 1, 1, 0, 0, 1]),
        rew=jnp.array([1.0, -2.0, 0.5, 3.0, -0.5, 0.0]),
        nobs=jnp.array([1, 0, 0, 1, 0, 1]),
    )
    rows = jnp.array([[0.9, 1, 1, 1], [0.999, 0.6, 10, 0.25]], dtype=jnp.float64)

    def train(row):
        return jax.lax.scan(
            lambda state, transition: (
                agent.update_with(transition, state, *row),
                None,
            ),
            initial,
            transitions,
        )[0]

    result = jax.jit(jax.vmap(train))(rows)
    for lane, row in enumerate(rows):
        eager = initial
        for index in range(6):
            transition = jax.tree.map(
                lambda value, index=index: value[index], transitions
            )
            eager = agent.update_with(transition, eager, *row)
        mapped = jax.tree.map(lambda value, lane=lane: value[lane], result)
        compiled = jax.jit(train)(row)
        for expected, actual, scalar in zip(
            jax.tree.leaves(eager),
            jax.tree.leaves(mapped),
            jax.tree.leaves(compiled),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(scalar, expected, rtol=2e-6, atol=2e-6)
        assert mapped.q.dtype == dtype
        assert mapped.velocity.dtype == dtype
    assert not np.array_equal(result.q[0], result.q[1])


def test_omitted_mapped_zeta_uses_the_component_setting():
    agent = NeSA(n_states=1, n_actions=1, gamma=0.9, zeta=0.25)
    initial = agent.init(jnp.zeros((1, 1)))
    actual = agent.update_with(sample(), initial, 0.9, 1.0, 1.0)
    np.testing.assert_array_equal(actual.q, [[0.25]])


@pytest.mark.parametrize("gamma", [0, 1, -1, float("nan")])
def test_invalid_discount(gamma):
    with pytest.raises(ValueError, match="gamma"):
        NeSA(n_states=2, n_actions=2, gamma=gamma)


@pytest.mark.parametrize("name", ["eta", "k0", "zeta"])
@pytest.mark.parametrize("value", [0, -0.1, float("inf"), float("nan")])
def test_invalid_hyperparameters(name, value):
    with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
        NeSA(n_states=2, n_actions=2, gamma=0.9, **{name: value})


@pytest.mark.parametrize("n_states,n_actions", [(0, 2), (2, 0)])
def test_invalid_dimensions(n_states, n_actions):
    with pytest.raises(ValueError, match="positive"):
        NeSA(n_states=n_states, n_actions=n_actions, gamma=0.9)


def test_invalid_shapes_and_dtypes():
    agent = NeSA(n_states=2, n_actions=3, gamma=0.9)
    with pytest.raises(AssertionError):
        agent.init(jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="floating"):
        agent.init(jnp.zeros((3, 2), dtype=jnp.int32))
    initial = agent.init(jnp.zeros((3, 2)))
    with pytest.raises(AssertionError):
        agent.update(replace(sample(), rew=jnp.ones(2)), initial)
    with pytest.raises(AssertionError):
        agent.update(sample(), replace(initial, velocity=jnp.zeros((2, 3))))
    with pytest.raises(AssertionError):
        agent.update_with(sample(), initial, 0.9, 1, 1, jnp.ones(2))
