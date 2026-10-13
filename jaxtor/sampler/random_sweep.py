""""Random sweep sampler for tabular MDPs"""

from __future__ import annotations

from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrd
import chex
from chex import dataclass
from jaxdp.mdp import MDP

Transition =TypeVar("Transition")

class Env(Protocol):
    class State(Protocol):
        mdp:MDP

class Mc(Protocol):
    class State(Protocol): ...
        
    def init(self, key: chex.PRNGKey, env: Env.State) -> Mc.State: ...
    def sample(
        self, act: chex.Array, state: Mc.State
        ) -> tuple[Transition, Mc.State]: ...
        
@dataclass
class RandomSweep:
    """Asynchronous random sampling without a trajectory.
    
    Flat bacth ordering: position = a*s +s (action-major)    
    """

    mc: Mc

    def _condition_mdp_initial(self, mdp: MDP, init_dist: chex.Array) -> MDP:
        """Create an MDP with a modified initial distribution.

        Args:
            mdp: Original MDP.
            init_dist: New initial state distribution (typically one-hot).

        Returns:
            MDP with the new initial distribution, sharing other arrays.
        """
        return MDP(
            transition=mdp.transition,
            reward=mdp.reward,
            initial=init_dist,
            terminal=mdp.terminal,
            features=mdp.features,
            name=mdp.name,
            validate=False,
        )
    
    def sample(self, key: chex.PRNGKey, env: Env.State) -> tuple[Transition, Mc.State]:
        """Sample one transition from given (s,a) pair
        
        Args: 
            key: Random key for initialization and sampling.
            enc: Environment state
            
        Returns:
            Transition and MC states
        """
        
        # Creating one hot vector for all (s,a) pairs with length (s)
        S, A = env.mdp.state_size, env.mdp.action_size
        state_indices = jnp.tile(jnp.arange(S), A)
        init_dists = jax.nn.one_hot(state_indices, S) # Size is (S*A, S)

        # Randomly selecting a one-hot vector for teh (s,a) pair
        new_key, s_key, a_key = jrd.split(key, 3)
        rand_s = jrd.randint(s_key, (), 0, S)
        rand_a = jrd.randint(a_key, (), 0, A)
        state_act_ind = rand_a*S + rand_s
        init_dist = init_dists[state_act_ind, :]
        chex.assert_shape(init_dist, (S,))

        def condition_env_state(init_dist: chex.Array) -> Env.State:
            new_mdp = self._condition_mdp_initial(env.mdp, init_dist)
            return env.replace(mdp=new_mdp)
        
        cond_env_states = condition_env_state(init_dist)

        mc_state = self.mc.init(new_key, cond_env_states)
        init_act = state_act_ind // S
        chex.assert_shape(init_act, ())
        return self.mc.sample(init_act, mc_state)

