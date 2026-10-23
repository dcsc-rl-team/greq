"""Behavior policies independent of the learner."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


@dataclass
class Uniform:
    """Sample uniformly from the action set using an independent random stream.

    Attributes:
        n_actions: Number of uniformly sampled actions.

    Public dataclasses:
        State: Behavior-policy random key.

    Public methods:
        init: Initialize the behavior random stream.
        act: Sample an action and advance the stream.
    """

    n_actions: int

    @dataclass
    class State:
        """Dynamic behavior state.

        Attributes:
            key: Random key, independent of the environment-step key.
        """

        key: jax.Array

    def __post_init__(self) -> None:
        """Require a nonempty action set."""
        if self.n_actions < 1:
            raise ValueError("n_actions must be positive")

    def init(self, key: jax.Array) -> Uniform.State:
        """Initialize the behavior-policy random stream."""
        return self.State(key=key)

    def act(
        self, obs: jax.Array, state: Uniform.State
    ) -> tuple[jax.Array, Uniform.State]:
        """Sample uniformly from the action set at a scalar observation."""
        chex.assert_rank(obs, 0)
        key, action_key = jax.random.split(state.key)
        action = jax.random.randint(action_key, (), 0, self.n_actions, dtype=jnp.int32)
        return action, self.State(key=key)
