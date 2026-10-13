from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pickle
import numpy as np
import glob
import os
import argparse
import pandas as pd
import seaborn as sns

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

def safe_log_plot_data(arr, floor=1e-15, ceiling=1e10):
    """
    Forces data to be within a plottable range for log scales.
    1. Converts to numpy.
    2. Replaces NaNs with the floor.
    3. Clips values to [floor, ceiling].
    """
    arr = np.array(arr)
    # Handle NaNs and Infs immediately
    arr = np.nan_to_num(arr, nan=floor, posinf=ceiling, neginf=floor)
    # Final clip to ensure no 0.0 or negative values reach the log-scale logic
    return np.clip(arr, a_min=floor, a_max=ceiling)

def plot_all_trials(alg_name, gamma):
    if not os.path.exists("Images"):
        os.makedirs("Images")

    search_pattern = f"{alg_name}_{gamma}_Trial_*_results.pkl"
    files = sorted(glob.glob(search_pattern))

    for file_path in files:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # Ensure steps are plottable on log-scale (no iteration 0)
        steps = np.array(data['steps'])
        steps = np.where(steps <= 0, 1, steps) 
        
        # Robustly preprocess metrics
        residuals = safe_log_plot_data(data['metrics']['bellman_linf'])
        trial_id = data.get('trial_id', data.get('trial_index', 'Unknown'))

        median_line = np.nanmedian(residuals, axis=0)
        lower_bound = np.nanpercentile(residuals, 25, axis=0)
        upper_bound = np.nanpercentile(residuals, 75, axis=0)

        plt.figure(figsize=(10, 6))
        
        for i in range(residuals.shape[0]):
            plt.plot(steps, residuals[i], color='gray', alpha=0.3, 
                     linewidth=0.8, label='Individual Seeds' if i == 0 else "")

        plt.fill_between(steps, lower_bound, upper_bound, 
                         color='blue', alpha=0.15, label='IQR (25th-75th %)')
        plt.plot(steps, median_line, color='darkblue', linewidth=2, 
                 label=f"Median (Trial {trial_id})")

        plt.xscale('log')
        plt.yscale('log')
        
           
        plt.title(f"{alg_name} | Trial {trial_id} (Robust BE Stats)")
        plt.xlabel("Iterations")
        plt.ylabel("Bellman Residual ($L_\infty$)")
        plt.grid(True, which="both", linestyle='--', alpha=0.4)
        plt.legend(loc='best')

        save_path = f"Images/{alg_name}_{gamma}_Trial_{trial_id}_BE_plot.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Generated robust BE plot: {save_path}")

def plot_all_trials_VE(alg_name, gamma):
    if not os.path.exists("Images"):
        os.makedirs("Images")

    search_pattern = f"{alg_name}_{gamma}_Trial_*_results.pkl"
    files = sorted(glob.glob(search_pattern))

    for file_path in files:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        steps = np.where(np.array(data['steps']) <= 0, 1, data['steps'])
        residuals = safe_log_plot_data(data['metrics']['value_norm'])
        trial_id = data.get('trial_id', data.get('trial_index', 'Unknown'))

        median_line = np.nanmedian(residuals, axis=0)
        lower_bound = np.nanpercentile(residuals, 25, axis=0)
        upper_bound = np.nanpercentile(residuals, 75, axis=0)

        plt.figure(figsize=(10, 6))
        
        for i in range(residuals.shape[0]):
            plt.plot(steps, residuals[i], color='gray', alpha=0.3, 
                     linewidth=0.8, label='Individual Seeds' if i == 0 else "")

        plt.fill_between(steps, lower_bound, upper_bound, 
                         color='blue', alpha=0.15, label='IQR (25th-75th %)')
        plt.plot(steps, median_line, color='darkblue', linewidth=2, 
                 label=f"Median (Trial {trial_id})")

        plt.xscale('log')
        plt.yscale('log')
        
        plt.title(f"{alg_name} | Trial {trial_id} (Robust VE Stats)")
        plt.xlabel("Iterations")
        plt.ylabel("Value error ($L_\infty$)")
        plt.grid(True, which="both", linestyle='--', alpha=0.4)
        plt.legend(loc='best')

        save_path = f"Images/{alg_name}_{gamma}_Trial_{trial_id}_VE_plot.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Generated robust VE plot: {save_path}")

def plot_all_trials_opt_rho(alg_name, gamma, opt_rho):
    if not os.path.exists("Images"):
        os.makedirs("Images")

    search_pattern = f"{alg_name}_{gamma}_Trial_*_results.pkl"
    files = sorted(glob.glob(search_pattern))

    for file_path in files:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # 1. FIX STEPS: Log scale cannot handle 0 or negative
        steps = np.array(data['steps'])
        steps = np.where(steps <= 0, 1, steps) 
        
        # 2. ROBUST DATA CLIPPING: Force values into a plottable log range
        # ZAP Q-Learning often diverges to 'inf', which causes the OverflowError
        raw_residuals = np.array(data['metrics']['pi_eval_rho'])
        
        # Replace NaNs/Infs and clip to a safe range [1e-15, 1e10]
        residuals = np.nan_to_num(raw_residuals, nan=1e-15, posinf=1e10, neginf=1e-15)
        residuals = np.clip(residuals, a_min=1e-15, a_max=1e10)

        trial_id = data.get('trial_id', data.get('trial_index', 'Unknown'))

        # NaN-safe statistics
        median_line = np.nanmedian(residuals, axis=0)
        lower_bound = np.nanpercentile(residuals, 25, axis=0)
        upper_bound = np.nanpercentile(residuals, 75, axis=0)

        plt.figure(figsize=(10, 6))
        
        for i in range(residuals.shape[0]):
            plt.plot(steps, residuals[i], color='gray', alpha=0.3, 
                     linewidth=0.8, label='Individual Seeds' if i == 0 else "")

        plt.fill_between(steps, lower_bound, upper_bound, 
                         color='blue', alpha=0.15, label='IQR (25th-75th %)')
        
        plt.plot(steps, median_line, color='darkblue', linewidth=2, 
                 label=f"Median (Trial {trial_id})")

        # 3. SAFETY FOR RHO LINE
        rho_val = max(opt_rho, 1e-15)
        plt.axhline(y=rho_val, color='red', linestyle='--', linewidth=1.5, 
                    label=f'Optimal $\\rho$ ({opt_rho:.4f})')

        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(bottom=1e-3, top = opt_rho*2)
        plt.title(f"{alg_name} | Trial {trial_id} (Robust GPE Stats)")
        plt.xlabel("Iterations")
        plt.ylabel("Greedy Policy Eval ($L_\infty$)")
        plt.grid(True, which="both", linestyle='--', alpha=0.4)
        plt.legend(loc='best', fontsize='small')

        save_path = f"Images/{alg_name}_{gamma}_Trial_{trial_id}_GPE_plot.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Generated robust GPE plot: {save_path}")



def plot_single_trial(alg_name, gamma, trial_id, output_dir="Images"):
    os.makedirs(output_dir, exist_ok=True)

    file_path = f"{alg_name}_{gamma}_Trial_{trial_id}_results.pkl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find trial file: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Ensure steps are plottable on log-scale (no non-positive values)
    steps = np.array(data['steps'])
    steps = np.where(steps <= 0, 1, steps)

    # Robustly preprocess metrics
    residuals = safe_log_plot_data(data['metrics']['bellman_linf'])
    actual_trial_id = data.get('trial_id', data.get('trial_index', trial_id))

    # Calculate summary statistics across seeds (axis 0)
    median_line = np.nanmedian(residuals, axis=0)
    lower_bound = np.nanpercentile(residuals, 25, axis=0)
    upper_bound = np.nanpercentile(residuals, 75, axis=0)

    plt.figure(figsize=(10, 6))

    # Plot individual runs
    for i in range(residuals.shape[0]):
        plt.plot(
            steps, residuals[i], 
            color='gray', alpha=0.3, linewidth=0.8, 
            label='Individual Seeds' if i == 0 else ""
        )

    # Plot IQR and Median
    plt.fill_between(
        steps, lower_bound, upper_bound, 
        color='blue', alpha=0.15, label='IQR (25th-75th %)'
    )
    plt.plot(
        steps, median_line, 
        color='darkblue', linewidth=2, 
        label=f"Median (Trial {actual_trial_id})"
    )

    plt.xscale('log')
    plt.yscale('log')

    plt.title(f"{alg_name} (γ={gamma}) | Trial {actual_trial_id} (Robust BE Stats)")
    plt.xlabel("Iterations")
    plt.ylabel("Bellman Residual ($L_\\infty$)")
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend(loc='best')

    save_path = os.path.join(output_dir, f"{alg_name}_{gamma}_Trial_{actual_trial_id}_BE_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated robust BE plot: {save_path}")

def plot_single_trial_VE(alg_name, gamma, trial_id, output_dir="Images"):
    os.makedirs(output_dir, exist_ok=True)

    file_path = f"{alg_name}_{gamma}_Trial_{trial_id}_results.pkl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find trial file: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Ensure steps are plottable on log-scale (no non-positive values)
    steps = np.array(data['steps'])
    steps = np.where(steps <= 0, 1, steps)

    # Robustly preprocess metrics
    residuals = safe_log_plot_data(data['metrics']['value_norm'])
    actual_trial_id = data.get('trial_id', data.get('trial_index', trial_id))

    # Calculate summary statistics across seeds (axis 0)
    median_line = np.nanmedian(residuals, axis=0)
    lower_bound = np.nanpercentile(residuals, 25, axis=0)
    upper_bound = np.nanpercentile(residuals, 75, axis=0)

    plt.figure(figsize=(10, 6))

    # Plot individual runs
    for i in range(residuals.shape[0]):
        plt.plot(
            steps, residuals[i], 
            color='gray', alpha=0.3, linewidth=0.8, 
            label='Individual Seeds' if i == 0 else ""
        )

    # Plot IQR and Median
    plt.fill_between(
        steps, lower_bound, upper_bound, 
        color='blue', alpha=0.15, label='IQR (25th-75th %)'
    )
    plt.plot(
        steps, median_line, 
        color='darkblue', linewidth=2, 
        label=f"Median (Trial {actual_trial_id})"
    )

    plt.xscale('log')
    plt.yscale('log')

    plt.title(f"{alg_name} (γ={gamma}) | Trial {actual_trial_id} (Robust VE Stats)")
    plt.xlabel("Iterations")
    plt.ylabel("Value Error ($L_\\infty$)")
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend(loc='best')

    save_path = os.path.join(output_dir, f"{alg_name}_{gamma}_Trial_{actual_trial_id}_VE_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated robust BE plot: {save_path}")

def plot_single_trial_opt_rho(alg_name, gamma, trial_id, opt_rho, output_dir="Images"):
    os.makedirs(output_dir, exist_ok=True)

    file_path = f"{alg_name}_{gamma}_Trial_{trial_id}_results.pkl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find trial file: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Ensure steps are plottable on log-scale (no non-positive values)
    steps = np.array(data['steps'])
    steps = np.where(steps <= 0, 1, steps)

    # Robustly preprocess metrics
    residuals = safe_log_plot_data(data['metrics']['pi_eval_rho'])
    actual_trial_id = data.get('trial_id', data.get('trial_index', trial_id))

    # Calculate summary statistics across seeds (axis 0)
    median_line = np.nanmedian(residuals, axis=0)
    lower_bound = np.nanpercentile(residuals, 25, axis=0)
    upper_bound = np.nanpercentile(residuals, 75, axis=0)

    plt.figure(figsize=(10, 6))

    # Plot individual runs
    for i in range(residuals.shape[0]):
        plt.plot(
            steps, residuals[i], 
            color='gray', alpha=0.3, linewidth=0.8, 
            label='Individual Seeds' if i == 0 else ""
        )

    # Plot IQR and Median
    plt.fill_between(
        steps, lower_bound, upper_bound, 
        color='blue', alpha=0.15, label='IQR (25th-75th %)'
    )
    plt.plot(
        steps, median_line, 
        color='darkblue', linewidth=2, 
        label=f"Median (Trial {actual_trial_id})"
    )

    rho_val = max(opt_rho, 1e-15)
    plt.axhline(y=rho_val, color='red', linestyle='--', linewidth=1.5, 
                label=f'Optimal $\\rho$ ({opt_rho:.4f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=1e-3, top = opt_rho*2)
    plt.title(f"{alg_name} | Trial {trial_id} (Robust GPE Stats)")
    plt.xlabel("Iterations")
    plt.ylabel("Greedy Policy Eval ($L_\infty$)")
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend(loc='best', fontsize='small')

    save_path = os.path.join(output_dir, f"{alg_name}_{gamma}_Trial_{actual_trial_id}_GPE_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated robust BE plot: {save_path}")

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_comparison(paths, labels, metric='bellman_linf', opt_rho=None):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Blue, Orange, Green, Red, Purple, Brown
    
    for path, label, color in zip(paths, labels, colors):
        try:
            data = load_data(path)
            values = np.array(data['metrics'][metric])
            steps = np.array(data['steps'])
            
            # Statistics
            median = np.nanmedian(values, axis=0)
            q1 = np.nanpercentile(values, 25, axis=0)
            q3 = np.nanpercentile(values, 75, axis=0)
            
            # 1. Plot individual seeds
            for i in range(values.shape[0]):
                plt.plot(steps, values[i], color=color, alpha=0.1, linewidth=0.5)
                
            # 2. Plot Median line
            plt.plot(steps, median, color=color, label=label, linewidth=2)
            
            # 3. Plot Shaded IQR
            plt.fill_between(steps, q1, q3, color=color, alpha=0.2)
            
        except Exception as e:
            print(f"Error loading {label}: {e}")

    # Metric-Specific Formatting
    if metric == 'pi_eval_rho':
        plt.ylabel(r'GPE', fontsize=25)
        plt.ylim(bottom = opt_rho*0.8)
        if opt_rho is not None:
            plt.axhline(y=opt_rho, color='red', linestyle='--', linewidth=2, 
                        label=f'Optimal $\\rho$ ({opt_rho})')
            
    elif metric == 'value_norm':
        plt.yscale('log')
        plt.xscale('log')
        # plt.ylim(top=1)
        plt.ylabel('VE', fontsize=25)
        
    elif metric == 'bellman_linf':
        plt.yscale('log')
        plt.xscale('log')
        # plt.ylim(bottom=5e-1, top=1)
        plt.ylabel(r'BE', fontsize=25)

    plt.xlabel('Samples', fontsize=25)
    # plt.title(f'Comparison: {metric.replace("_", " ").title()}', fontsize=25)
    plt.legend(loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0.0,
                frameon=True,fontsize=20)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    # plt.ylim(bottom=8)
    output_name = f"Comparison_Plot_{metric}.png"
    plt.savefig(output_name, dpi=300)
    print(f"Plot saved as {output_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precond', type=str, required=True)
    parser.add_argument('--qlearn', type=str, required=True)
    parser.add_argument('--zap', type=str, required=True)
    parser.add_argument('--zap_od', type=str, required=True)
    parser.add_argument('--sql', type=str, required=True)
    parser.add_argument('--nesa', type=str, required=True)
    # parser.add_argument('--zap_zero', type=str, required=True)
    parser.add_argument('--metric', type=str, default='bellman_linf', 
                        choices=['bellman_linf', 'value_norm', 'pi_eval_rho'])
    parser.add_argument('--opt_rho', type=float, help="Optimal rho value for pi_eval_rho")

    args = parser.parse_args()
    paths = [args.precond, args.qlearn, args.zap, args.zap_od, args.sql, args.nesa]
    labels = ['GReQ', 'QL', 'ZAP QL', 'ZAP QL $\mathcal{O}(d)$', 'SQL', 'NeSA']
    
    plot_comparison(paths, labels, metric=args.metric, opt_rho=args.opt_rho)

if __name__ == "__main__":
    main()

