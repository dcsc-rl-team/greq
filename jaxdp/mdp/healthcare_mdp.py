import jax.numpy as jnp

from jaxdp.mdp import MDP

def healthcare_mdp() -> MDP:
    """
    Constructs Healthcare MDP for treatment and intensity of doses.

    - State space: Health condition (1 to 6, 1 being the best and 6 the worst, which is absorbing mortality state).
    - Actions:
         Dosage Level (1 to 3, 1 being low dosage, and 3 being high dosage)
    - Rewards:
         For s in {1,2,3,4,5}, a in {1,2,3}:
            r(s,a) = \sum_{s^+=1}^{6} s^+ + P(s^+ |s,a) + a
            and
            r(6,:) = 0
    - Initial state: Uniformaly distributed over the states.
    - Terminal: state 6 is the mortality state, which is absorbing state making it the terminal state.

    Returns:
        MDP: The constructed Forest MDP.
    """  
    n_states = 6
    n_actions = 3

    transition = jnp.zeros((n_actions, n_states, n_states))
    transition = transition.at[0].set(jnp.array([
                                  [0.7, 0.3, 0, 0, 0, 0],
                                  [0.3, 0.4, 0.3, 0, 0, 0],
                                  [0, 0.3, 0.4, 0.3, 0, 0],
                                  [0, 0, 0.3, 0.4, 0.3, 0],
                                  [0, 0, 0, 0.3, 0.4, 0.3],
                                  [0, 0, 0, 0, 0, 1]]))
    
    transition = transition.at[1].set(jnp.array([
                                  [0.8, 0.2, 0, 0, 0, 0],
                                  [0.4, 0.4, 0.2, 0, 0, 0],
                                  [0, 0.4, 0.4, 0.2, 0, 0],
                                  [0, 0, 0.4, 0.4, 0.2, 0],
                                  [0, 0, 0, 0.4, 0.4, 0.2],
                                  [0, 0, 0, 0, 0, 1]]))
    
    transition = transition.at[2].set(jnp.array([
                                  [0.9, 0.1, 0, 0, 0, 0],
                                  [0.5, 0.4, 0.1, 0, 0, 0],
                                  [0, 0.5, 0.4, 0.1, 0, 0],
                                  [0, 0, 0.5, 0.4, 0.1, 0],
                                  [0, 0, 0, 0.5, 0.4, 0.1],
                                  [0, 0, 0, 0, 0, 1]]))
    
    transition = transition.transpose(0, 2, 1) # P is now column stochatic matrix for each action

    states = jnp.array([1,2,3,4,5,6]) # Actual numerical value
    actions = jnp.array([1,2,3]) # Actual numerical value

    reward = jnp.zeros((n_actions, n_states, n_states))
    for i in range(n_actions):
        for j in range(n_states-1):
            for k in range(n_states-1):
                reward = reward.at[i,j,k].set(jnp.einsum("s,s->", states, transition[i,:,k])+ actions[i])         
    

    initial = jnp.ones((n_states,)) / n_states # pdf
    terminal = jnp.zeros(n_states).at[5].set(1.0) # marker vector
    
    return MDP(transition, reward, initial, terminal, name=f"HealthCareMDP")
