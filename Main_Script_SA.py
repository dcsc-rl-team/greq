from __future__ import annotations
from typing import Any
import argparse

import dataclasses
import time

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import rlax 
import tyro
from chex import dataclass
from rich import print
from rich.progress import track

from jaxdp.mdp import MDP as JaxdpMDP
from tabular_env import TabularEnv, garnet, graph, gridworld
from jaxtor.eval.tabular import Eval as Evaluator, optimal_q
from jaxtor.sampler import Imc, Mc, random_sweep
from Algorithms import q_learning_async, speedy_q_learning_async, nesa_q_learning_async, zap_q_learning_async, zap_q_learning_od_async, zap_zero_q_learning_async, pre_cond_q_learning_async
from utils_SA import plot_results, plot_VE_results, plot_opt_rho_results, benchmark_plot_results, benchmark_plot_results_VE

jax.config.update("jax_enable_x64", True)

@dataclass
class Agent:
    """Agent is the policy. \epsilon-greedy tabular agent with an (A, S) array."""

    epsilon: float # decaying value

    @dataclass
    class State:
        key: chex.Array
        alg_state: Any

    @dataclass
    class Args:
        value_fn: Any
   
    def q_vals(self, state: Agent.State, obs: chex.Array) -> chex.Array:
        """Q-values for given state indices."""
        return state.alg_state.q_vals[:, obs]
    
    def act(
        self, obs: chex.Array, state: Agent.State
    ) -> tuple[chex.Array, Agent.State]:
        """ε-greedy action selection."""
        key, act_key, explore_key = jrd.split(state.key, 3)
        greedy = jnp.argmax(self.q_vals(state, obs))
        random = jrd.randint(act_key, (), 0, state.alg_state.q_vals.shape[0])
        action = jnp.where(jrd.uniform(explore_key) < self.epsilon, random, greedy)
        return action, state.replace(key=key)

   
@jax.jit
def sample_env_step(state: Imc.State) -> tuple[Any, Imc.State]:
    """Isolate just the environment sampling logic.
    'imc' is captured safely from the outer scope as a closure.
    """
    trans, state = imc.sample(state)
    return trans, state

@jax.jit(static_argnames=("agent_args_value_fn",))
def update_alg_step(alg_state: Any, trans: Any, alpha: float, alpha_power: float, beta_power: float, k: int, agent_args_value_fn: Any) -> Any:
    """Isolate strictly the mathematical update of the Q-values/algorithm state."""
    return agent_args_value_fn.update(alg_state, trans, alpha, alpha_power, beta_power, k)

@jax.jit(static_argnames=("agent_args_value_fn",))
def rand_train_step(state: Agent.State, env_state: JaxdpMDP, alpha_power, beta_power, k, agent_args_value_fn, key) -> Agent.State:
    """Random-access transition + Q-learning update with decaying step size."""
 
    key, sampler_key = jrd.split(key, 2)
    trans, mc_state = ran_sweep.sample(sampler_key, env_state) 
    new_env_state = mc_state.env
    alpha = cfg.alpha_init / (1.0 + k / cfg.alpha_period) ** cfg.alpha_power
    new_alg_state = agent_args_value_fn.update(state.alg_state, trans, alpha, alpha_power, beta_power, k)
    return state.replace(alg_state=new_alg_state), key, new_env_state
    

######################################################################################

####################################### LOOP INPUTS ##################################

@dataclass 
class Config:
    """Training configuration for tyro CLI"""

    benchmark_type: str 
    garnet: garnet.Config = dataclasses.field(default_factory=garnet.Config) # I want to make this an input. Ex: alg_MDP is the input, it gets the MDP from here
    graph: graph.Config = dataclasses.field(default_factory=graph.Config) # I want to make this an input. Ex: alg_MDP is the input, it gets the MDP from here
    gridworld: gridworld.Config = dataclasses.field(
    default_factory=lambda: gridworld.Config(
        board = [
                        "#####",
                        "#  @#",
                        "# # #",
                        "#P  #",
                        "#####"
                    ],
        p_slip=0.25
    )
)
    n_steps: int = 1_000_000
    alpha_init: float = 1
    alpha_power: float = 0.8
    alpha_period: float = 1
    beta_power: float = 0.7
    alpha_min: float = 0.0001
    warmup_steps: int = 5_000
    gamma: float = 0.999
    epsilon: float = 0.1
    eval_freq: int = 1_000
    seed: int = 0

cfg = tyro.cli(Config)

######################################################################################

############################### ALGORITHM IMPLEMENTATION #############################

def alg_implementation(imc_state, agent_args, eval_state, opt_q, opt_rho, alg_name, alpha_power, beta_power):

    print(f"[bold green]{alg_name}")
    results = {}
    t0 = time.time()
    itr_t = 0.0
    
    for k in track(range(cfg.n_steps), description="Training"):
        # CALL IT LIKE THIS NOW:
        trans, imc_state = sample_env_step(imc_state)
        
        # Calculate step decay parameters
        alpha = cfg.alpha_init / (2 + k / cfg.alpha_period) ** cfg.alpha_power

        # B. Start timing strictly the algorithm math block
        itr_t0 = time.time()
        
        new_alg_state = update_alg_step(
            imc_state.agent.alg_state, 
            trans, 
            alpha, 
            alpha_power, 
            beta_power, 
            k, 
            agent_args.value_fn
        )
        
        # C. CRITICAL: Force execution to finish on the device to get true hardware timing
        new_alg_state.q_vals.block_until_ready()
        
        # D. Calculate execution duration for this specific update
        itr_time = time.time() - itr_t0
        itr_t = itr_t + itr_time
        
        # E. Reconstruct the state tree for the next loop iteration
        imc_state = imc_state.replace(agent=imc_state.agent.replace(alg_state=new_alg_state))

        # Evaluation Block
        if (k + 1) % cfg.eval_freq == 0:
            m, eval_state = jit_eval(eval_state, opt_q, imc_state.agent)
            print(
                f"  step {k + 1:6d}"
                f"  bellman={float(m.bellman_linf):.4f}"
                f"  value={float(m.value_norm):.4f}"
                f"  ρ(π)={float(m.pi_eval_rho):.3f}"
            )
            results[k+1] = m

    elapsed = time.time() - t0
    print(
        f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s"
        f"\n[bold green]Iteration Time ONLY!:[/bold green] {itr_t:.4f}s"
        f"  value_norm={float(m.value_norm):.6f}"
        f"  bellman_linf={float(m.bellman_linf):.6f}"
        f"  ρ*(π)={opt_rho:.3f}"
    )
    print(f" Count of state-action pair = {imc_state.agent.alg_state.c}")
    plot_results(results, alg_name)
    plot_VE_results(results, alg_name)
    return results

######################################################################################

############################ ALGORITHM IMPLEMENTATION RAND ###########################

def alg_implementation_rand(agent_state, env_state, agent_args, eval_state, opt_q, opt_rho, alg_name, key, alpha_power, beta_power,):

    print(f"[bold green]{alg_name}")
    results = {}
    t0 = time.time()
    for k in track(range(cfg.n_steps), description="Training"):
        agent_state, key, env_state = rand_train_step(agent_state, env_state, alpha_power, beta_power, k, agent_args.value_fn, key)

        if (k + 1) % cfg.eval_freq == 0: 
            m, eval_state = jit_eval(eval_state, opt_q, agent_state)        
            print(
                f"  step {k + 1:6d}"
                f"  bellman={float(m.bellman_linf):.4f}"
                f"  value error={float(m.value_norm):.4f}"
                f"  ρ(π)={float(m.pi_eval_rho):.3f}"
            )
            
            results[k+1] = m

    elapsed = time.time() - t0
    print(
        f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s"
        f"  value error={float(m.value_norm):.6f}"
        f"  bellman_linf={float(m.bellman_linf):.6f}"
        f"  ρ*(π)={opt_rho:.3f}"
    )
    print(f" Count of state-action pair = {agent_state.alg_state.c}")
    plot_results(results, alg_name)
    plot_VE_results(results, alg_name)
    plot_opt_rho_results(results, alg_name)
    return results

######################################################################################

######################### ALGORITHM IMPLEMENTATION RAND VMAP #########################

def alg_implementation_rand_vmap(agent_state, env_state, agent_args, eval_state, opt_q, opt_rho, alg_name, key):

    print(f"[bold green]{alg_name}")
    results = {}
    t0 = time.time()
    for k in track(range(cfg.n_steps), description="Training"):
        agent_state, key, env_state = rand_train_step(agent_state, env_state, k, agent_args.value_fn, key)

        if (k + 1) % cfg.eval_freq == 0: 
            m, eval_state = jit_eval(eval_state, opt_q, agent_state)        
            print(
                f"  step {k + 1:6d}"
                f"  bellman={float(m.bellman_linf):.4f}"
                f"  value error={float(m.value_norm):.4f}"
                f"  ρ(π)={float(m.pi_eval_rho):.3f}"
            )
            
            results[k+1] = m

    elapsed = time.time() - t0
    print(
        f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s"
        f"  value error={float(m.value_norm):.6f}"
        f"  bellman_linf={float(m.bellman_linf):.6f}"
        f"  ρ*(π)={opt_rho:.3f}"
    )
    print(f" Count of state-action pair = {agent_state.alg_state.c}")
    plot_results(results, alg_name)
    plot_VE_results(results, alg_name)
    plot_opt_rho_results(results, alg_name)
    return results

######################################################################################

############################## BENCHMARK IMPLEMENTATION  #############################

def benchmark_rand(env_state, opt_q, opt_rho, key):

    alg_map = {
        q_learning_async: "Q-Learning (Random)",
        zap_q_learning_async: "ZAP Q-Learning (Random)",
        pre_cond_q_learning_async: "Pre-Cond Q-Learning (Random)",
        }
    all_results = {}

    for alg_module, alg_name in alg_map.items():

        agent_args = Agent.Args(value_fn=alg_module)
        init_alg_state = agent_args.value_fn.init(env_state.mdp, cfg.gamma)
        agent_state = Agent.State(key=agent_key, alg_state=init_alg_state)
        eval_state = evaluator.init(agent_state)
 
        results = alg_implementation_rand(agent_state, env_state, agent_args, eval_state, opt_q, opt_rho, alg_name, key)
        all_results[alg_name] = results
        print(f"Completed for {alg_name}")

    benchmark_plot_results(all_results, cfg.gamma, cfg.eval_freq)
    benchmark_plot_results_VE(all_results, cfg.gamma, cfg.eval_freq)
    return all_results

######################################################################################

if __name__ == "__main__":
    cfg = tyro.cli(Config)

    # Map the CLI input to (Algorithm Module, Sampling Type, Display Name)
    # sampling_type: 0 for trajectory (imc), 1 for sweep
    alg_map = {
        "q_learning_imc": (q_learning_async, 0, "Q-Learning_(Trajectory)"),
        "q_learning_rand": (q_learning_async, 1, "Q-Learning_(Random)"),
        "speedy_ql_imc": (speedy_q_learning_async, 0, "SQL_(Trajectory)"),
        "speedy_ql_rand": (speedy_q_learning_async, 1, "SQL_(Random)"),
        "nesa_ql_imc": (nesa_q_learning_async, 0, "NeSA_(Trajectory)"),
        "nesa_ql_rand": (nesa_q_learning_async, 1, "NeSA_(Random)"),
        "zap_ql_imc": (zap_q_learning_async, 0, "ZAP_Q-Learning_(Trajectory)"),
        "zap_ql_rand": (zap_q_learning_async, 1, "ZAP_Q-Learning_(Random)"),
        "zap_ql_od_imc": (zap_q_learning_od_async, 0, "ZAP_Q-Learning_Od_(Trajectory)"),
        "zap_ql_od_rand": (zap_q_learning_od_async, 1, "ZAP_Q-Learning_Od_(Random)"),
        "zap_zero_ql_imc": (zap_zero_q_learning_async, 0, "ZAP_Zero_Q-Learning_(Trajectory)"),
        "zap_zero_ql_rand": (zap_zero_q_learning_async, 1, "ZAP_Zero_Q-Learning_(Random)"),
        "pre_cond_ql_imc": (pre_cond_q_learning_async, 0, "Pre-Cond_Q-Learning_(Trajectory)"),
        "pre_cond_ql_rand": (pre_cond_q_learning_async, 1, "Pre-Cond_Q-Learning_(Randoms)"),
        "all_rand": (benchmark_rand, 1, "Full Benchmark")
    }

    if cfg.benchmark_type not in alg_map:
        raise ValueError(f"Unknown Benchmark type: {cfg.benchmark_type}. "
                         f"Available: {list(alg_map.keys())}")

    alg_module, sampling_type, alg_name = alg_map[cfg.benchmark_type]

    agent = Agent(epsilon=cfg.epsilon)
    env=garnet.make(cfg.garnet)

    

    if sampling_type == 1:

        mc_sampler = Mc(
                    max_episode_len=cfg.graph.max_episode_len,
                    queue_size=20,
                    env=env,
                    )
        ran_sweep = random_sweep.RandomSweep(mc=mc_sampler)
           
        key = jrd.PRNGKey(cfg.seed)
        key, env_key, agent_key = jrd.split(key, 3)
        env_state = env.init(env_key)
    
        opt_q = optimal_q(env_state.mdp, cfg.gamma)
        opt_rho = float(jnp.sum(env_state.mdp.initial * jnp.max(opt_q, axis=0)))
    
        evaluator = Evaluator(mdp=env_state.mdp, gamma=cfg.gamma, agent=agent)
        jit_eval = jax.jit(evaluator.metric)
        if alg_module == benchmark_rand:
            benchmark_rand(env_state, opt_q, opt_rho, key)
        else:
            agent_args = Agent.Args(value_fn=alg_module)
            init_alg_state = agent_args.value_fn.init(env_state.mdp, key, cfg.gamma)
            agent_state = Agent.State(key=agent_key, alg_state=init_alg_state)
            eval_state = evaluator.init(agent_state)
    
            alg_implementation_rand(agent_state, env_state, agent_args, eval_state, opt_q, opt_rho, alg_name, key, cfg.alpha_power, cfg.beta_power)


    else:

        imc = Imc(
        agent=agent,
        mc=Mc(
            max_episode_len=cfg.graph.max_episode_len,
            queue_size=20,
            env=env,
            ),
        )
           
        key = jrd.PRNGKey(cfg.seed)
        key, env_key, agent_key = jrd.split(key, 3)
        env_state = imc.mc.env.init(env_key)

        opt_q = optimal_q(env_state.mdp, cfg.gamma)
        opt_rho = float(jnp.sum(env_state.mdp.initial * jnp.max(opt_q, axis=0)))

        evaluator = Evaluator(mdp=env_state.mdp, gamma=cfg.gamma, agent=agent)
        jit_eval = jax.jit(evaluator.metric)
        agent_args = Agent.Args(value_fn=alg_module)
        init_alg_state = agent_args.value_fn.init(env_state.mdp, cfg.gamma)
        agent_state = Agent.State(key=agent_key, alg_state=init_alg_state)
        imc_state = imc.init(mc=imc.mc.init(agent_key, env_state), agent=agent_state)
        eval_state = evaluator.init(agent_state)

        alg_implementation(imc_state, agent_args, eval_state, opt_q, opt_rho, alg_name, cfg.alpha_power, cfg.beta_power)