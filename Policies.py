import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from Tools import e_greedy_policy
from jaxdp.typehints import F, PiType, QType, StaticMeta

class epsilon_greedy(metaclass=StaticMeta):
    r"""
    Epsilon greedy exploration policy

    Selects random action with probability \epsilon and greedy action otherwise.
    \epsilon decay over time: epsilon=max(epsilon*decay, min)

    """

    @struct.dataclass
    class State:
        epsilon: F[""] # Exploration rate (scalar)
        eps_decay: F[""] # Decay factor (scalar)
        eps_min: F[""] # Minimum epsilon (scalar)

    def init(epsilon: float, eps_decay: float, eps_min: float) -> "epsilon_greedy.State":
        epsilon = 1.0
        eps_decay = 0.997
        eps_min = 0.1

        return epsilon_greedy.State(epsilon=jnp.array(epsilon), eps_decay=jnp.array(eps_decay), eps_min=jnp.array(eps_min))
    
    def update(state: "epsilon_greedy.State", done: F[""]) -> "epsilon_greedy.State":
        """
        Decay \epsilon after each episode
        """
        new_eps = jnp.maximum(state.epsilon*state.eps_decay, state.eps_min)
        epsilon = jnp.where(done, new_eps, state.epsilon)
        return state.replace(epsilon=epsilon)
    
    def get_policy(q_vals: QType, state: "epsilon_greedy.State") -> PiType:
        """
        \epsilon-greedy policy is computed using q_vals, and \epsilon values
        """
        return e_greedy_policy.q(q_vals, state.epsilon)

    
class soft_policy(metaclass=StaticMeta):
    r"""
    Soft Policy (Boltzmann Exploration)

    Selects actions with probability proportional to exp(Q(s,a)/temperature).
    Temperature decays over time: temp = max(temp * decay, min)    
    """

    @struct.dataclass
    class State:
        temperature: F[""] # Temp parameter (scalar)
        temp_decay: F[""] # Decay factor (scalar)
        temp_min: F[""] # Min. Temp (scalar)

    def init(temperature: float, temp_decay: float, temp_min: float) -> "soft_policy.State":
        temperature = 1.0
        temp_decay = 0.995
        temp_min = 0.01

        return soft_policy.State(temperature = jnp.array(temperature), temp_decay = jnp.array(temp_decay), temp_min=jnp.array(temp_min))
    
    def update(state: "soft_policy.State", done: F[""]) -> "soft_policy.State":
        """
        Temperature decay after each episode
        """

        new_temp = jnp.maximum(state.temperature*state.temp_decay, state.temp_min)
        temp = jnp.where(done, new_temp, state.temperature)
        return state.replace(temperature = temp)
    
    def get_policy(q_vals: QType, state: "soft_policy.State") -> PiType:
        """
        Get soft policy from q-values
        """
        scaled_q = q_vals/state.temperature
        exp_q = jnp.exp(scaled_q - jnp.max(scaled_q, axis=0, keepdims = False))
        policy = exp_q/jnp.sum(exp_q, axis=0, keepdims=True)
        return policy
    
   