from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp
import jax.random as jrd
import chex
from chex import dataclass
from jaxdp.mdp import MDP as JaxdpMDP
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.simple_graph import graph_mdp as jaxdp_graph_mdp
from jaxdp.mdp.grid_world import grid_world
from jaxdp.mdp.healthcare_mdp import healthcare_mdp
from jaxdp.mdp.forest_mdp import forest_mdp

def _sample_transition(
        key: chex.PRNGKey, mdp: JaxdpMDP, s: chex.Numeric, a: chex.Numeric
) -> tuple[chex.Numeric, chex.Numeric, chex.Numeric]:
    """Sample transition from MDP using indices.
    
    Args:
        key: Random key.
        mdp: MDP with transition[A, S', S] and reward[A, S, S'].
        s: Current state index.
        a: Action index.
        
    Returns:
        [next_state, reward, term] tuple.
    """
    probs = mdp.transition[a,:,s]
    s_next = jrd.choice(key, mdp.state_size, p=probs)
    rew = mdp.reward[a,s,s_next]
    term = mdp.terminal[s_next]
    return s_next, rew, term

class ConfigProtocol(Protocol):
    """Defines Protocol for MDP configurations. 
    
    By inheriting from Protocol, instructing Python type checker (like Mypy or Pyright) that this 
    class isn't meant to be instantiated. Instead, it defines a contract. Any other class that has a 
    max_episode_len attribute and an init_mdp method is automatically considered a "ConfigProtocol," even if it 
    doesn't explicitly inherit from it.
    """

    max_episode_len: int # Number of steps an episode lasts before its truncated.

    def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP: ... # In a Protocol, you don't write the actual code logic. You just define the signature.

@dataclass 
class TabularEnv:
    """Index-based tabular MDP environmetn.
    
    Attributes: 
        config: MDP configuration following ConfigProtocol.
    """

    @dataclass
    class State:
        """ Environment State.

        Attributes:
            mdp: Underlying jaxdp MDP instance.
            s: Current state index.
            step: Current step within the epsiode
            max_episode_len: Maximum episode length before truncation.
        """

        mdp: JaxdpMDP
        s: chex.Numeric
        step: chex.Numeric
        max_episode_len: chex.Numeric

    @dataclass
    class Step:
        """Single step transition result.
        
        Attributes:
            nobs: Next observation (state index).
            rew: Reward.
            term: Termination state.
            trun: Truncation Flag.
        """

        nobs: chex.Numeric
        rew: chex.Numeric
        term: chex.Numeric
        trun: chex.Numeric

    config: ConfigProtocol

    def step(
            self, key: chex.PRNGKey, act: chex.PRNGKey, state: State
            ) -> tuple[Step, State]:
        
        s_next, rew, term = _sample_transition(key, state.mdp, state.s, act)
        trun = state.step >= state.max_episode_len - 1
        new_state = state.replace(s=s_next, step = state.step+1)
        return TabularEnv.Step(nobs=s_next, rew=rew, term=term, trun=trun), new_state
    
    def init(self, key: chex.PRNGKey) -> TabularEnv.State:
        """Ïnitializing the environmetn state.
        
        Arguments:
            key: Random key for MDP initialization.
            
        Returns:
            Initialized State.
        """
        mdp = self.config.init_mdp(key)
        return TabularEnv.State(
            mdp=mdp,
            s=jnp.array(-1),
            step = jnp.array(0),
            max_episode_len=jnp.array(self.config.max_episode_len),)
    
    def obs(self, state: TabularEnv.State) -> chex.Numeric: # why state: TabularEnv.State and not just State?
        """Get observation from state.

        Args:
            state: Current state.

        Returns:
            State index.
        """
        return state.s
    
    def reset(
        self, key: chex.PRNGKey, state: TabularEnv.State
    ) -> tuple[chex.Numeric, TabularEnv.State]:
        """Reset to a new episode.

        Args:
            key: Random key for sampling initial state.
            state: Current state.

        Returns:
            Initial observation and reset state.
        """
        s = jrd.choice(key, state.mdp.state_size, p=state.mdp.initial)
        new_state = state.replace(s=s, step=jnp.array(0))  # type: ignore[attr-defined]
        return (s, new_state)
    
class garnet:
    """Garnet MDP namespace."""

    @dataclass
    class Config:
        """Configuration for creating a Garnet MDP.

        Attributes:
            state_size: Number of states in the MDP.
            action_size: Number of actions available.
            branch_size: Number of successor states per state-action pair.
            min_reward: Minimum reward value.
            max_reward: Maximum reward value.
            max_episode_len: Maximum episode length before truncation.
        """

        state_size: int = 30
        action_size: int = 15
        branch_size: int = 5
        min_reward: float = 0.0
        max_reward: float = 1.0
        max_episode_len: int = 1000

        def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP:
            """Ïnitialize a garnet MDP from this config."""
            return garnet_mdp(
                state_size=self.state_size,
                action_size=self.action_size,
                branch_size=self.branch_size,
                min_reward=self.min_reward,
                max_reward=self.max_reward,
                key=key,
            )
    
    @staticmethod
    def make(config: garnet.Config) -> TabularEnv:
        """Create a Garnet tabular environment.

        Args:
            config: Garnet MDP configuration.

        Returns:
            TabularEnv instance.
        """
        return TabularEnv(config=config)

class graph:
    """Graph MDP namespace."""

    @dataclass
    class Config:
        """"Configuration for creating a graph MDP.
        
        Attributes:
            max_epsiode_len: Maximum epsiode length before truncation.
        """

        max_episode_len: int = 10000
        
        def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP:
            """Initializes a graph mdp for this config."""
            return jaxdp_graph_mdp()
    
    @staticmethod
    def make(config: graph.Config) -> TabularEnv:
        """Create a graph tabular environment
        
        Attributes:
            config: Graph MDP configuration.
            
        Returns:
            TabularEnv instance.
        """
        return TabularEnv(config=config)

class gridworld:
    """GridWorld MDP namespace."""

    @dataclass
    class Config:
        """Configuration for creating a GridWorld MDP.

        Board characters:
            '#': Impassable wall
            'P': Initial agent position
            '@': Terminal/goal state (positive reward)
            '=': Absorbing state (positive reward)
            '+': Positive reward cell
            'X': Penalty cell
            ' ': Regular passable space

        Attributes:
            board: List of strings representing the 2D grid layout.
            p_slip: Probability of slipping to unintended action.
            max_episode_len: Maximum episode length before truncation.

        Example:
            >>> config = gridworld.Config(
            ...     board=[
            ...         "#####",
            ...         "#  @#",
            ...         "# #X#",
            ...         "#P  #",
            ...         "#####"
            ...     ],
            ...     p_slip=0.1
            ... )
        """

        board: list[str] 
        
        p_slip: float = 0.25
        max_episode_len: int = 1000

        def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP:
            """Initialize a GridWorld MDP from this config."""
            return grid_world(board=self.board, p_slip=self.p_slip)

    @staticmethod
    def make(config: gridworld.Config) -> TabularEnv:
        """Create a GridWorld tabular environment.

        Args:
            config: GridWorld MDP configuration.

        Returns:
            TabularEnv instance.
        """
        return TabularEnv(config=config)   
    
class healthcare:
    """Healtcare MDP namespace."""

    @dataclass
    class Config:
        """"Configuration for creating a healthcare MDP.
        
        Attributes:
            max_episode_len: Maximum episode length before truncation.
        """

        max_episode_len: int = 1000
        
        def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP:
            """Initialze healthcare MDP from this config."""
            return healthcare_mdp()
        
    @staticmethod
    def make(config: healthcare.Config) -> TabularEnv:
        """Create a Healthcare tabular Environment.
        
        Args:
            config: Healthcare MDP configuration.
            
        Returns: 
            TabularEnv instance.
        """
        return TabularEnv(config=config)
    
class forest:
    """Forest MDP namespace."""

    @dataclass
    class Config:
        """Configuration for creating a forest MDP.
        
        Attributes:
            max_episode_len: Maximum epsiode length before truncation.
        """
        rotation: int = 25
        max_epsiode_len: int = 1000
        def init(self, key: chex.PRNGKey) -> JaxdpMDP:
            """Initialize Forest MDP for this config."""
            return forest_mdp(rotation=self.rotation)

    @staticmethod
    def make(config: forest.Config) -> TabularEnv:
        """"Create a tabular environment.
        
        Args:
            config: Forest MDP configuration.
            
        Returns:
            TabularEnv instance.
        """ 
        return TabularEnv(config=config)
       