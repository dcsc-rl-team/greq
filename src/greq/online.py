"""Composition of an independent behavior sampler and an online learner."""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


@dataclass
class Uniform:
    """Sample actions without consulting the learner.

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


class Sampler[Sample, S](Protocol):
    """Transition sampling capability consumed by :class:`Online`."""

    def sample(self, state: S, /) -> tuple[Sample, S]: ...


class Learner[Sample, S](Protocol):
    """Update capability consumed by :class:`Online`."""

    def update(self, transition: Sample, state: S, /) -> S: ...


@dataclass
class Online[Sample, SamplerS, LearnerS]:
    """Advance sampling and learning without storing per-step matrix states.

    Required protocols::

        sampler.sample(sampler_state) -> (transition, sampler_state)
        learner.update(transition, learner_state) -> learner_state

    Attributes:
        sampler: Transition sampler with its own random streams.
        learner: Learning component, independent of behavior selection.
        n_steps: Number of transitions per call to ``advance``.

    Public dataclasses:
        State: Child states and completed sample count.

    Public methods:
        init: Combine initialized child states.
        advance: Run one fixed-size block of online updates.
    """

    sampler: Sampler[Sample, SamplerS]
    learner: Learner[Sample, LearnerS]
    n_steps: int

    @dataclass
    class State[SamplerData, LearnerData]:
        """Dynamic experiment state.

        Attributes:
            sampler: Transition sampler state.
            learner: Learning state.
            step: Total number of completed transitions.
        """

        sampler: SamplerData
        learner: LearnerData
        step: jax.Array

    def __post_init__(self) -> None:
        """Reject empty blocks."""
        if self.n_steps < 1:
            raise ValueError("n_steps must be positive")

    def init(
        self, sampler: SamplerS, learner: LearnerS
    ) -> Online.State[SamplerS, LearnerS]:
        """Combine child states without initializing them implicitly."""
        return self.State(
            sampler=sampler, learner=learner, step=jnp.array(0, jnp.int32)
        )

    def advance(
        self, state: Online.State[SamplerS, LearnerS]
    ) -> Online.State[SamplerS, LearnerS]:
        """Run a block, retaining only its final state."""
        return jax.lax.fori_loop(0, self.n_steps, self._step, state)

    def _step(
        self, index: jax.Array, state: Online.State[SamplerS, LearnerS]
    ) -> Online.State[SamplerS, LearnerS]:
        """Sample one transition and pass it to the learner."""
        transition, sampler = self.sampler.sample(state.sampler)
        learner = self.learner.update(transition, state.learner)
        return self.State(sampler=sampler, learner=learner, step=state.step + 1)
