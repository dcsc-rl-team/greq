"""Paper-compatible Garnet rewards with jaxdp transition generation."""

from __future__ import annotations

from dataclasses import dataclass as static_dataclass
from math import isfinite

import jax
import jax.numpy as jnp
import numpy as np
from jaxdp.mdp.garnet import garnet_mdp
from jaxtor.env.tabular import Mdp


@static_dataclass(frozen=True)
class Garnet:
    """Host-side configuration consumed by Jaxtor's ``TabularEnv``.

    Transition matrices come from ``jaxdp``. Rewards are independent uniform
    draws per state-action pair, broadcast over successor states to implement
    the paper's deterministic ``r(s, a)`` convention.
    """

    n_states: int = 10
    n_actions: int = 5
    branching: int = 5
    min_reward: float = 0.0
    max_reward: float = 1.0
    max_eps_len: int = 2_147_483_647

    def __post_init__(self) -> None:
        """Reject invalid sizes, reward intervals, and episode limits."""
        if self.n_states < 1 or self.n_actions < 1:
            raise ValueError("state and action counts must be positive")
        if not 1 <= self.branching <= self.n_states:
            raise ValueError("branching must lie between 1 and n_states")
        if not (
            isfinite(self.min_reward)
            and isfinite(self.max_reward)
            and self.min_reward < self.max_reward
        ):
            raise ValueError("reward bounds must be finite and strictly increasing")
        if not 1 <= self.max_eps_len <= 2_147_483_647:
            raise ValueError("max_eps_len must fit a positive int32")

    def init_mdp(self, key: jax.Array) -> Mdp:
        """Generate dynamics and deterministic rewards from separate subkeys."""
        transition_key, reward_key = jax.random.split(key)
        generated = garnet_mdp(
            state_size=self.n_states,
            action_size=self.n_actions,
            branch_size=self.branching,
            key=transition_key,
        )
        reward = jax.random.uniform(
            reward_key,
            (self.n_actions, self.n_states),
            minval=self.min_reward,
            maxval=self.max_reward,
        )
        return Mdp(
            transition=generated.transition,
            reward=jnp.broadcast_to(reward[:, :, None], generated.reward.shape),
            initial=generated.initial,
            terminal=generated.terminal,
        )


def require_irreducible(mdp: Mdp) -> None:
    """Reject missing state coverage under uniform behavior, without resampling."""
    reachable = np.any(np.asarray(mdp.transition) > 0, axis=0).T
    reachable |= np.eye(mdp.state_size, dtype=bool)
    for k in range(mdp.state_size):
        reachable |= reachable[:, k, None] & reachable[None, k, :]
    if not reachable.all():
        raise ValueError("uniform behavior is not irreducible for this MDP seed")
