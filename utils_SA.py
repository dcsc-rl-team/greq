from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_results(results, alg_name):
    
    iterations, values = zip(*sorted(results.items()))
    bellman_errors = [m.bellman_linf for m in values]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, bellman_errors, label=f"{alg_name} (Bellman $L_\infty$)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(alg_name)
    plt.xlabel("Iterations")
    plt.ylabel("Bellman Residual")
    plt.ylim(top=1, bottom=1e-5)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.savefig(f"Images/{alg_name}_BE_plot.png", dpi=300)
    plt.close()

def plot_VE_results(results, alg_name):
    
    iterations, values = zip(*sorted(results.items()))
    value_errors = [m.value_norm for m in values]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, value_errors, label=f"{alg_name} ($VE$)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(alg_name)
    plt.xlabel("Iterations")
    plt.ylabel("Value Errors")
    plt.ylim(top=1e3, bottom=1e-2)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.savefig(f"Images/{alg_name}_VE_plot.png", dpi=300)
    plt.close()

def plot_opt_rho_results(results, alg_name):

    iterations, values = zip(*sorted(results.items()))
    greddy_policy_eval = [m.pi_eval_rho for m in values]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, greddy_policy_eval, label=f"{alg_name} ($VE$)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(alg_name)
    plt.xlabel("Iterations")
    plt.ylabel("Greedy policy Value")
    plt.ylim(top=1e3, bottom=1e-2)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.savefig(f"Images/{alg_name}_GVE_plot.png", dpi=300)
    plt.close()


def benchmark_plot_results(all_results, gamma, eval_freq):
    
    plt.figure(figsize=(10, 6))
    for alg_name, data_dict in all_results.items():
        sorted_steps = sorted(data_dict.keys())
        y_values = [float(data_dict[step].bellman_linf) for step in sorted_steps]
        plt.plot(sorted_steps, y_values, label=alg_name)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Benchmark")
    plt.xlabel("Iterations")
    plt.ylabel("Bellman Residual")
    plt.ylim(top=1e2, bottom=1e-3)
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.savefig(f"Images/Benchmark_BE_{gamma}_plot.png", dpi=300)
    plt.close()

def benchmark_plot_results_VE(all_results, gamma, eval_freq):
    
    plt.figure(figsize=(10, 6))
    for alg_name, data_dict in all_results.items():
        sorted_steps = sorted(data_dict.keys())
        y_values = [float(data_dict[step].value_linf) for step in sorted_steps]
        plt.plot(sorted_steps, y_values, label=alg_name)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title(alg_name)
    plt.xlabel("Iterations")
    plt.ylabel("Value Errors")
    plt.ylim(top=1e3, bottom=1e-2)
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.savefig(f"Images/Benchmark_VE_{gamma}_plot.png", dpi=300)
    plt.close()