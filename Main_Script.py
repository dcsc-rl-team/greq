from __future__ import annotations
from typing import Any
import argparse

import dataclasses
import time

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import pickle
import numpy as np
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
from utils import plot_results, plot_all_trials, plot_VE_results, plot_all_trials_VE, plot_all_trials_opt_rho, plot_opt_rho_results, plot_single_trial, plot_single_trial_VE, plot_single_trial_opt_rho

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

def to_pure_dict(obj: Any) -> Any:
    """Recursively converts any Pytree, Dataclass, or Metrics object into a pure nested dict."""
    if hasattr(obj, "_asdict"):  # Handles NamedTuples or Metrics classes
        return {k: to_pure_dict(v) for k, v in obj._asdict().items()}
    elif dataclasses.is_dataclass(obj):
        return {f.name: to_pure_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    elif hasattr(obj, "__dict__"): # Generic objects
        return {k: to_pure_dict(v) for k, v in vars(obj).items() if not k.startswith('_')}
    elif isinstance(obj, dict):
        return {k: to_pure_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_pure_dict(v) for v in obj]
    else:
        return obj
       
@jax.jit(static_argnames=("agent_args",))
def train_step(state: Imc.State, k: int, agent_args, alpha_init, alpha_period, alpha_power, beta_power) -> Imc.State:
    """One transition + Q-learning update with decaying step size."""
 
    trans, state = imc.sample(state)
    alpha = alpha_init / (2 + k / alpha_period) ** alpha_power
    new_alg_state = agent_args.value_fn.update(state.agent.alg_state, trans, alpha, alpha_power, beta_power, k) #Experimenting with learning rate for PCQL
    return state.replace(agent=state.agent.replace(alg_state=new_alg_state))

@jax.jit(static_argnames=("agent_args_value_fn",))
def rand_train_step(state: Agent.State, env_state: JaxdpMDP, k, agent_args_value_fn, key, alpha_init, alpha_period, alpha_power, beta_power) -> Agent.State: # Remove beta_power for QL 
    """Random-access transition + Q-learning update with decaying step size."""
 
    key, sampler_key = jrd.split(key, 2) # When batched, the key values are different for each seed, giving different results even if initialization is same
    trans, mc_state = ran_sweep.sample(sampler_key, env_state) 
    new_env_state = mc_state.env
    alpha = alpha_init / (1.0 + k / alpha_period) ** alpha_power
    new_alg_state = agent_args_value_fn.update(state.alg_state, trans, alpha, alpha_power, beta_power, k) # Remove beta_power for QL  
    return state.replace(alg_state=new_alg_state), key, new_env_state
    

######################################################################################

####################################### LOOP INPUTS ##################################

@dataclass 
class Config:
    """Training configuration for tyro CLI"""

    benchmark_type: str 
    garnet: garnet.Config = dataclasses.field(default_factory=garnet.Config) 
    graph: graph.Config = dataclasses.field(default_factory=graph.Config) 
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

    ##########################################################################################################################################
    #### SINGLE ####
    # alpha_period: float = 1
    # alpha_power: float = 1
    # beta_power: float = 0.9 # QL
    ################
    
    ############# RANDOM SAMPLING #################
    # QL(16)
    # alpha_period: Any = dataclasses.field(
    #     default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0]
    # )    
    # alpha_power: Any = dataclasses.field(
    #     default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7]
    # )
    # beta_power: float = 0.9 # QL
    
    # Pre and Zap (40)
    # alpha_period: Any = dataclasses.field(
    #     default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
    #                              10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 
    #                              100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 
    #                              1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    # )
    
    # alpha_power: Any = dataclasses.field(
    #     default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.7, 
    #                              1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.7,
    #                              1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.7,
    #                              1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.7,
    #                              ]
    # )

    # beta_power: Any = dataclasses.field(
    #     default_factory=lambda: [0.9, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7, 0.6, 0.6, 
    #                              0.9, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7, 0.6, 0.6,
    #                              0.9, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7, 0.6, 0.6,
    #                              0.9, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7, 0.6, 0.6,
    #                              ]
    # )

    ################################################
    ##########################################################################################################################################

    ################## MC SAMPLING #################
    # QL MC sampling (24)
    # alpha_period: Any = dataclasses.field(
    #     default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    # )
    
    # alpha_power: Any = dataclasses.field(
    #     default_factory=lambda: [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # )

    # beta_power: float = 0.9 # QL

    # SQL MC Sampling (1)
    alpha_period: float = 1
        
    alpha_power: float = 1
    
        
    beta_power: float = 1

    # Pre and Zap MC sampling (10)
    # alpha_period: float = 1
    
    # alpha_power: Any = dataclasses.field(
    #     default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.7
    #                              ]
    # )

    
    # beta_power: Any = dataclasses.field(
    #     default_factory=lambda: [0.9, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7, 0.6, 0.6
    #                              ]
    # )
    ################################################

    alpha_min: float = 0.0001
    warmup_steps: int = 5_000
    gamma: float = 0.9
    epsilon: float = 1
    eval_freq: int = 1_000 
    seed: int = 0
    q_val_seed: int = 0
    n_seed: int = 10

cfg = tyro.cli(Config)
alpha_power_array = jnp.array(cfg.alpha_power)
alpha_period_array = jnp.array(cfg.alpha_period)
beta_power_array = jnp.array(cfg.beta_power)

######################################################################################

############################ ALGORITHM IMPLEMENTATION RAND ###########################

def alg_implementation_rand(agent_state, env_state, agent_args, eval_state, opt_q, opt_rho, alg_name, key, alpha_init, alpha_period, alpha_power, beta_power): # Remove beta_power for QL 

    print(f"[bold green]{alg_name}")
    results = {}
    t0 = time.time()
    for k in track(range(cfg.n_steps), description="Training"):
        agent_state, key, env_state = rand_train_step(agent_state, env_state, k, agent_args.value_fn, key, alpha_init, alpha_period, alpha_power, beta_power) # Remove beta_power for QL 

        if (k + 1) % cfg.eval_freq == 0: 
            m, eval_state = jit_eval(eval_state, opt_q, agent_state)                    
            results[k+1] = m

    elapsed = time.time() - t0
    return results

######################################################################################

############################ ALGORITHM IMPLEMENTATION MC ###########################

def alg_implementation_mc(imc_state, agent_args, eval_state, opt_q, opt_rho, alg_name, alpha_init, alpha_period, alpha_power, beta_power): # Remove beta_power for QL 

    print(f"[bold green]{alg_name}")
    results = {}
    # t0 = time.time()
    for k in track(range(cfg.n_steps), description="Training"):
        imc_state = train_step(imc_state, k, agent_args, alpha_init, alpha_period, alpha_power, beta_power) # Remove beta_power for QL 

        if (k + 1) % cfg.eval_freq == 0:
            m, eval_state = jit_eval(eval_state, opt_q, imc_state.agent)
            results[k+1] = m

    # elapsed = time.time() - t0
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

    # benchmark_plot_results(all_results, cfg.gamma, cfg.eval_freq)
    # benchmark_plot_results_VE(all_results, cfg.gamma, cfg.eval_freq)
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
    env=garnet.make(cfg.garnet) # Change MDP name here

    

    if sampling_type == 1:

        mc_sampler = Mc(
                    max_episode_len=cfg.garnet.max_episode_len,
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
            keys = jrd.split(jrd.PRNGKey(cfg.q_val_seed), cfg.n_seed)
            # keys = jrd.PRNGKey(cfg.q_val_seed)
            agent_args = Agent.Args(value_fn=alg_module)
            vmap_init = jax.vmap(agent_args.value_fn.init, in_axes = (None, 0, None)) 
            init_alg_state = vmap_init(env_state.mdp, keys, cfg.gamma) # Batched initial state
            vmap_agent = jax.vmap(lambda k, s: Agent.State(key=k, alg_state=s), in_axes = (None, 0))
            agent_state = vmap_agent(agent_key, init_alg_state) # Batched agent state
            vmap_eval = jax.vmap(evaluator.init, in_axes=(0))
            eval_state = vmap_eval(agent_state) # Batched evaluator state
            vmap_seeds = jax.vmap(alg_implementation_rand, 
                                    in_axes=(0, None, None, 0, None, None, None, 0, None, None, None, None)) # 
            vmap_trials = jax.vmap(vmap_seeds, 
                                    in_axes=(None, None, None, None, None, None, None, None, None, None, 0, 0)) # Remove beta_power for QL , 0
            # vmap_trials = vmap_seeds
            all_results = vmap_trials(agent_state, env_state, agent_args, eval_state, opt_q, opt_rho, alg_name, keys, cfg.alpha_init, cfg.alpha_period, alpha_power_array, beta_power_array) # Remove beta_power for QL 
            
            results_cpu = jax.device_get(all_results)
            sample_val = list(results_cpu.values())[0]
            if dataclasses.is_dataclass(sample_val):
                all_keys = [f.name for f in dataclasses.fields(sample_val)]
            else:
                # Fallback for standard objects
                all_keys = [k for k in vars(sample_val).keys() if not k.startswith('_')]

            structured_results = {}

            for key in all_keys:
                # Stack the metrics from every timestep into one big array
                # We use getattr() to pull the specific metric (e.g., bellman_linf) from each Metrics object
                metric_series = jnp.stack(
                    [getattr(results_cpu[t], key) for t in sorted(results_cpu.keys())], 
                    axis=-1
                )
                structured_results[key] = metric_series

            for i in range(40): # make 16 for QL
                trial_data = {
                    "metrics": {k: v[i] for k, v in structured_results.items()},
                    "alpha_power": cfg.alpha_power[i],
                    "alpha_period": cfg.alpha_period[i],
                    "beta_power": cfg.beta_power[i], # Remove beta_power for QL
                    "steps": list(sorted(results_cpu.keys())),
                    "trial_id": i + 1
                }

                filename = f"{alg_name}_{cfg.gamma}_Trial_{i+1}_results.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(trial_data, f)
                print(f"Successfully saved {filename}")
            
            plot_all_trials(alg_name, cfg.gamma)
            plot_all_trials_VE(alg_name, cfg.gamma)
            plot_all_trials_opt_rho(alg_name, cfg.gamma, opt_rho)

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
        key, env_key, agent_key_bs = jrd.split(key, 3)
        agent_key = jrd.split(agent_key_bs, cfg.n_seed)
        env_state = imc.mc.env.init(env_key)

        opt_q = optimal_q(env_state.mdp, cfg.gamma)
        opt_rho = float(jnp.sum(env_state.mdp.initial * jnp.max(opt_q, axis=0)))

        evaluator = Evaluator(mdp=env_state.mdp, gamma=cfg.gamma, agent=agent)
        jit_eval = jax.jit(evaluator.metric)

        # keys = jrd.split(jrd.PRNGKey(cfg.q_val_seed), cfg.n_seed)
        agent_args = Agent.Args(value_fn=alg_module)

        # vmap_init = jax.vmap(agent_args.value_fn.init, in_axes = (None, 0, None)) 
        # init_alg_state = vmap_init(env_state.mdp, keys, cfg.gamma) # Batched initial state

        init_alg_state = agent_args.value_fn.init(env_state.mdp, cfg.gamma)

        vmap_agent = jax.vmap(lambda k, s: Agent.State(key=k, alg_state=s), in_axes = (0, None))
        agent_state = vmap_agent(agent_key, init_alg_state) # Batched agent state for different action

        vmap_mc_init = jax.vmap(imc.mc.init, in_axes=(0, None))
        vmap_imc_init = jax.vmap(imc.init, in_axes=(0, 0))

        mc_state = vmap_mc_init(agent_key, env_state)
        imc_state = vmap_imc_init(mc_state, agent_state)

        vmap_eval = jax.vmap(evaluator.init, in_axes=0)
        eval_state = vmap_eval(agent_state) # Batched evaluator state
        vmap_seeds = jax.vmap(alg_implementation_mc, 
                                in_axes=(0, None, 0, None, None, None, None, None, None, None)) # 
        # vmap_trials = jax.vmap(vmap_seeds, 
        #                         in_axes=(None, None, None, None, None, None, None, None, 0, 0)) # Remove beta_power for QL , 0
        vmap_trials = vmap_seeds # For single hyperparameter
        all_results = vmap_trials(imc_state, agent_args, eval_state, opt_q, opt_rho, alg_name, cfg.alpha_init, cfg.alpha_period, cfg.alpha_power, cfg.beta_power) # Remove beta_power for QL 
        
        results_cpu = jax.device_get(all_results)
        sample_val = list(results_cpu.values())[0]
        if dataclasses.is_dataclass(sample_val):
            all_keys = [f.name for f in dataclasses.fields(sample_val)]
        else:
            # Fallback for standard objects
            all_keys = [k for k in vars(sample_val).keys() if not k.startswith('_')]

        structured_results = {}

        for key in all_keys:
            # Stack the metrics from every timestep into one big array
            # We use getattr() to pull the specific metric (e.g., bellman_linf) from each Metrics object
            metric_series = jnp.stack(
                [getattr(results_cpu[t], key) for t in sorted(results_cpu.keys())], 
                axis=-1
            )
            structured_results[key] = metric_series

        #########VARIOUS HYPERPARAMETERS#########
        # for i in range(10): # make 16 for QL
        #     trial_data = {
        #         "metrics": {k: v[i] for k, v in structured_results.items()},
        #         "alpha_power": cfg.alpha_power[i],
        #         "alpha_period": cfg.alpha_period[i],
        #         "beta_power": cfg.beta_power[i], # Remove beta_power for QL
        #         "steps": list(sorted(results_cpu.keys())),
        #         "trial_id": i + 1
        #     }

        #     filename = f"{alg_name}_{cfg.gamma}_Trial_{i+1}_results.pkl"
        #     with open(filename, "wb") as f:
        #         pickle.dump(trial_data, f)
        #     print(f"Successfully saved {filename}")
        
        # plot_all_trials(alg_name, cfg.gamma)
        # plot_all_trials_VE(alg_name, cfg.gamma)
        # plot_all_trials_opt_rho(alg_name, cfg.gamma, opt_rho)

        #########SQL#########
        trial_data = {
            "metrics": structured_results,  # was: {k: v[i] for k, v in structured_results.items()}
            "steps": list(sorted(results_cpu.keys())),
            "trial_id": 1
        }

        # CHANGE 2: Hardcode Trial 1 (or 1 instead of {i+1})
        filename = f"{alg_name}_{cfg.gamma}_Trial_1_results.pkl"
        with open(filename, "wb") as f:
            pickle.dump(trial_data, f)
        print(f"Successfully saved {filename}")

        plot_single_trial(alg_name, cfg.gamma, 1, 'Images')
        plot_single_trial_VE(alg_name, cfg.gamma, 1, 'Images')
        plot_single_trial_opt_rho(alg_name, cfg.gamma, 1, opt_rho, 'Images')