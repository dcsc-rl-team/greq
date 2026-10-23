"""Uniform behavior preserves its explicit random stream under JAX transforms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from greq.policy import Uniform


def test_uniform_actions_follow_the_explicit_random_stream():
    policy = Uniform(n_actions=4)
    key = jax.random.key(7)
    state = policy.init(key)
    act = jax.jit(policy.act)
    for _ in range(20):
        key, action_key = jax.random.split(key)
        expected = jax.random.randint(action_key, (), 0, 4, dtype=jnp.int32)
        action, state = act(jnp.array(0), state)
        np.testing.assert_array_equal(action, expected)
        np.testing.assert_array_equal(
            jax.random.key_data(state.key), jax.random.key_data(key)
        )


def test_uniform_policy_is_observation_independent_and_vmappable():
    policy = Uniform(n_actions=3)
    keys = jax.random.split(jax.random.key(8), 3)
    states = jax.vmap(policy.init)(keys)
    actions, after = jax.jit(jax.vmap(policy.act))(jnp.arange(3), states)
    for index in range(3):
        expected, state = policy.act(jnp.array(0), policy.init(keys[index]))
        np.testing.assert_array_equal(actions[index], expected)
        np.testing.assert_array_equal(
            jax.random.key_data(after.key[index]), jax.random.key_data(state.key)
        )


@pytest.mark.parametrize("size", [0, -1])
def test_uniform_policy_requires_actions(size):
    with pytest.raises(ValueError, match="positive"):
        Uniform(n_actions=size)


def test_uniform_policy_requires_a_scalar_observation():
    policy = Uniform(n_actions=2)
    with pytest.raises(AssertionError):
        policy.act(jnp.array([0]), policy.init(jax.random.key(0)))
