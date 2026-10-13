import chex
import distrax
import jax
import jax.numpy as jnp
import jax.random as jrd

from jaxdp.mdp.mdp import MDP
from jaxdp.typehints import F, QType, VType, PiType, StaticMeta

class Bellman_Policy_Operator(metaclass=StaticMeta):

    def v(mdp: MDP, policy: PiType, value: VType, gamma: float) -> VType:
        r""""
        Bellman operator evaluates the policy for given state

        .. math::
            \mathcal{T}_pi v(s) = \sum_a \pi(a|s) \left[ r(s,a) + \gamma \sum_{s^+} P(s^+|s,a)v(s^+) \right]

        Args:
            mdp (MDP): Markopv Decision Process
            policy (PiType): Policy Distribution \pi(a|s)
            value (VType): State value array
            gamma (float): Discount Factor

        Return: 
            VType: Value function on which Bellman policy operator has been applied

        """

        target_values = jnp.einsum("axs,x,x->as",
                                   mdp.transition, value, (1-mdp.terminal)) # (1 - mdp.terminal) considered for making out terminal states as they have zero value mdp.terminal is 1 hot vector.
        
        exp_reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)

        return jnp.einsum("as,as->s", policy, exp_reward + gamma*target_values)

    def q(mdp: MDP, policy: PiType, value: QType, gamma: float) -> QType:
        r"""
        Bellman policy operator applied to state-action pair

        .math::
            \mathcal{T}_pi q(s) =  r(s,a) + \gamma \sum_{s^+} P(s^+|s,a) \sum_a \pi(a|s) q(s^+) 

        Args:
            mdp (MDP): Transition probability 
            policy (PiType): policy distribution \pi(a|s)
            value (QType): Q_value array
            gamma (float): Discount Factor

        Return:
            QType: Return bellman policy operator applied to input Q_value
        """        

        target_value = jnp.einsum("axs,ux,ux,x->as", 
                                  mdp.transition, policy, value, (1-mdp.transition))
        
        exp_reward = jnp.einsum("asx,axs", mdp.reward, mdp.transition)

        return exp_reward + gamma*target_value
    

class Bellman_Optimality_Operator(metaclass=StaticMeta):

    def v(mdp: MDP, value: VType, gamma: float) -> VType:
        r"""
        Bellman optimality operator applied on V-function

        .math::
            \mathcal{T} v(s) = \max_a  \left[ r(s,a) + \gamma \sum_{s^+} P(s^+|s,a)v(s^+) \right]

        Args:
            mdp (MDP): Transition probability 
            value (VType): state value function
            gamma (float): Discount Factor

        Returns:
            State value function with bellman optimality operator applied on it 
        """

        target_values = jnp.einsum("axs,x->as", mdp.transition, value)

        exp_reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)

        return jnp.max(exp_reward + gamma*target_values, axis = 0, keepdims = False)

    def q(mdp: MDP, value: QType, gamma: float) -> QType:
        r"""
        Bellman optimality operator applied on Q-function

        .math::
            \mathcal{T} q(s) = r(s,a) + \gamma \sum_{s^+} P(s^+|s,a) \max_a q(s^+)

        Args:
            mdp (MDP): Transition probability 
            value (QType): state-action value function
            gamma (float): Discount Factor

        Returns:
            State-action value function with bellman optimality operator applied on it 
        """

        target_values = jnp.einsum("axs,x->as", mdp.transition, 
                                   jnp.max(value, axis = 0, keepdims = False))

        exp_rewards = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)

        return exp_rewards + gamma*target_values

class greedy_policy(metaclass=StaticMeta):

    def v(mdp: MDP, value: VType, gamma: float) -> PiType:
        r"""
                Greedy policy distribution from state values.
        
        .math::
            \pi_v(s,a) = \arg_max_{a \in A}[r(s,a) + gamma* \sum_{s^+ \in S} P(s^+|s,a)v(s^+)]

        Args:
            mdp (MDP): Markov Decision Process
            value (VType): State value array
            gamma (float): Discount factor

        Returns:
            PiType: Policy distribution as one hot vectors
        """
        target_values = jnp.einsum("axs,x->as", mdp.transition, value)

        exp_reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        q_sa = exp_reward + gamma*target_values

        return jax.nn.one_hot(jnp.argmax(q_sa, axis = 0, keepdims = False), 
                              num_classes=q_sa.shape[0],
                              axis=0)

        # return greedy_policy.q(to_state_action_value(mdp, value, gamma))
    
    def q(value: QType) -> PiType:
        """
        Greedy policy distribution from Q values.

        .math::
            \pi_q(s,a) = \arg_max_{a \in A}(q(s,a))

        Args:
            value (QType): Q Value array

        Returns:
            PiType: Policy distribution as one hot vectors

        """

        
        return jax.nn.one_hot(jnp.argmax(value, axis=0, keepdims=False), 
                              num_classes=value.shape[0],
                              axis=0)
    

class soft_policy(metaclass=StaticMeta):

    def q(value: QType, temperature: float) -> PiType:
        r"""
        Softmax Policy Distribution
        
        Args: 
            value (QType): Q Value array
            temperature (float): This is the temperature parameter of doftmx function. 
                Lower the value of temperature results in a uniform policy distribution 
                and a higher vlaue will result in a distribution closer to greedy policy.
                
        Retruns:
            PiType: Policy distribution as one hot vectors

        """
        return jax.nn.softmax(value/temperature, axis = 0)

    def v(value: VType, temperature: float) -> PiType:
        raise NotImplementedError
    
class e_greedy_policy(metaclass=StaticMeta):

    def q(value: QType, epsilon: float) -> PiType:
        r""""
        Epsilon greedy policy distribution

        Args: 
            value (QType): Q Value array
            temperature (float): Epsilon parameter. 
                Policy takes greedy action with (1 - \epsilon + \epsilon/|A|) probability, and the 
                exploration action is taken with probability \epsilon/|A| where |A| is teh dimension of teh action space.
                
        Retruns:
            PiType: Policy distribution as one hot vectors

        """
        greedy_p = greedy_policy.q(value)
        return greedy_p*(1-epsilon) + jnp.ones_like(value)*(epsilon/value.shape[0])

    def v(value: VType, epsilon: float, ) -> PiType:
        raise NotImplementedError
    

class policy_evaluation(metaclass=StaticMeta):

    def v(mdp: MDP, policy: PiType, gamma: float) -> VType:
        r"""
        Evaluate the Policy using the MDP

        ..math::
            \eta(\pi)(s_i) = \big[(\mathrm{I}_n - \gamma P^\pi)r^\pi\big]_i

        Args:
            mdp (MDP): Markov decision Process
            policy (PiType): Policy Distribution
            gamma (float): Discount Factor

        Returns:
            VType: Cumulative discounted reward of each state        
        """
        transition_pi, reward_pi = _markov_chain_pi(mdp, policy)

        return (jnp.linalg.inv(jnp.eye(mdp.state_size) - gamma * transition_pi.T) @ 
                jnp.einsum("sx, sx->s", transition_pi.T, reward_pi))

    def q(mdp: MDP, policy: PiType, gamma: float) -> QType:
        r"""
        Evaluate the policy for each state-action pair using the true MDP

        .. math::
            \\q(\\pi)(s_i, a_j) = \\big[r^{a_j} + \\gamma P^\\pi
            (\\mathrm{I}_n - \\gamma P^\\pi)r^\\pi\\big]_i

        Args:
            mdp (MDP): Markov Decision Process
            policy (PiType): Policy distribution
            gamma (float): Discount factor

        Returns:
            QType: Cumulative discounted reward of each state-action pair
        
        """
        mc_state_values = policy_evaluation.v(mdp, policy, gamma)
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        return (reward + gamma * jnp.einsum("axs,x->as", mdp.transition, mc_state_values))


def _markov_chain_pi(mdp: MDP, policy: PiType) -> tuple[F["SS"], F["SS"]]:
    r"""
    Given a policy for an MDP, make Markov Chain

    ..math::
        P^\pi = \underset{a \sim \pi}{\mathbb{E}}[P^a]
        r^\pi = \underset{a \sim \pi}{\mathbb{E}}[r^a]

    Args: 
        mdp (MDP): Markopv Decision Process
        policy (PiType): Policy Distribution    
    
    Returns:
        tuple[chex.Array, chex.Array]: Transition matrix, Reward matrix         
    
    """
    transition_pi = jnp.einsum("as,axs->xs", policy, mdp.transition)
    reward_pi = jnp.einsum("as,asx->sx", policy, mdp.reward)
    return transition_pi, reward_pi

def to_state_action_value(mdp: MDP, value: VType, gamma: float) -> QType:
    """Convert state values to Q-values using the MDP dynamics."""
    # TODO: Add test
    return (jnp.einsum("asx,axs->as", mdp.reward, mdp.transition) +
            gamma * jnp.einsum("axs,x->as", mdp.transition, value))

def sync_sample(mdp: MDP, key: chex.PRNGKey) -> tuple[F["AS"], F["ASS"], F["AS"]]:
    r"""
    Synchronously sample starting from state action pair in the given MDP

    Args: 
        mdp (MDP): Markov decision Process
        key (chex.PRNGKey): JAX Pseudorandom number generator (PRNGs)

    Returns:
        tuple[chex.Array, chex.Array, chex.Array]: Reward, Next state, Termination condition

    """
    next_state = distrax.OneHotCategorical(probs = jnp.einsum("axs->asx", mdp.transition), 
                                           dtype = "float").sample(seed=key) # The output of this will have the shape (A,S,S^+), one hot encoding in the third axes direction
    
    terminal = jnp.einsum("asx,x->as", next_state, mdp.terminal)
    reward = jnp.einsum("asx,asx->as", mdp.reward, next_state)

    return reward, next_state, terminal

def exp_decay(step: int) -> F[""]:
    
    temperature = 1.0
    offset = 1
    decay = 1/(step/temperature + offset)
    
    return decay 
    

