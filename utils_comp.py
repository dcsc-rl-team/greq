import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_multi_metric_comparison(paths, labels, metrics, opt_rho=None, output_name="Comparison_Multi_Metric.png"):
    # Clean seaborn grid style matching publication plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    num_metrics = len(metrics)
    
    # 1. Create a row of subplots
    fig, axes = plt.subplots(1, num_metrics, figsize=(6.5 * num_metrics, 5.5), sharex=False)
    if num_metrics == 1:
        axes = [axes]

    # Preload data once so we don't read from disk repeatedly per metric
    loaded_data = []
    for path, label in zip(paths, labels):
        try:
            loaded_data.append((label, load_data(path)))
        except Exception as e:
            print(f"Error loading {label} from {path}: {e}")
            loaded_data.append((label, None))

    # 2. Plot data on each subplot
    for ax, metric in zip(axes, metrics):
        for idx, (label, data) in enumerate(loaded_data):
            if data is None or metric not in data.get('metrics', {}):
                continue
            
            color = colors[idx % len(colors)]
            values = np.array(data['metrics'][metric])
            steps = np.array(data['steps'])

            median = np.nanmedian(values, axis=0)
            q1 = np.nanpercentile(values, 25, axis=0)
            q3 = np.nanpercentile(values, 75, axis=0)

            # Individual runs / seeds
            for i in range(values.shape[0]):
                ax.plot(steps, values[i], color=color, alpha=0.1, linewidth=0.6)

            # Median line (only label on the first axis to keep handles clean)
            ax.plot(steps, median, color=color, label=label, linewidth=2)

            # IQR shading
            ax.fill_between(steps, q1, q3, color=color, alpha=0.2)

        # Metric-specific formatting
        ax.set_xlabel('Steps', fontsize=16)
        ax.grid(True, which="both", ls="-", alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)

        if metric == 'pi_eval_rho':
            ax.set_ylabel(r'Policy Evaluation $\rho$', fontsize=16)
            if opt_rho is not None:
                ax.axhline(y=opt_rho, color='red', linestyle='--', linewidth=1.5,
                           label=f'Optimal $\\rho$ ({opt_rho})')
                ax.set_ylim(bottom=opt_rho * 0.8)
                
        elif metric == 'value_norm':
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel('Value Norm (Log Scale)', fontsize=16)
            
        elif metric == 'bellman_linf':
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r'Bellman $L_\infty$ Error (Log Scale)', fontsize=16)

        ax.set_title(metric.replace("_", " ").title(), fontsize=18, pad=10)

    # 3. Collect handles & labels from the first subplot
    handles, labels_list = axes[0].get_legend_handles_labels()

    # 4. Create common legend placed outside above the subplots
    fig.legend(
        handles=handles,
        labels=labels_list,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05), # Centered horizontally, pushed above subplots
        ncol=len(handles),          # Keep all items in 1 single horizontal row
        frameon=False,              # No bounding box
        fontsize=14,
        columnspacing=1.5,
        handletextpad=0.5
    )

    # 5. Leave room for the elevated top legend
    plt.tight_layout()
    fig.subplots_adjust(top=0.82)

    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {output_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precond', type=str, required=True)
    parser.add_argument('--qlearn', type=str, required=True)
    parser.add_argument('--zap', type=str, required=True)
    parser.add_argument('--zap_od', type=str, required=True)
    parser.add_argument('--sql', type=str, required=True)
    parser.add_argument('--nesa', type=str, required=True)
    parser.add_argument('--opt_rho', type=float, default=None, help="Optimal rho value for pi_eval_rho")

    args = parser.parse_args()
    paths = [args.precond, args.qlearn, args.zap, args.zap_od, args.sql, args.nesa]
    labels = ['GReQ', 'QL', 'ZAP QL', r'ZAP QL $\mathcal{O}(d)$', 'SQL', 'NeSA']
    
    # Plot all 3 metrics side by side
    metrics_to_plot = ['bellman_linf', 'value_norm', 'pi_eval_rho']
    plot_multi_metric_comparison(paths, labels, metrics=metrics_to_plot, opt_rho=args.opt_rho)

if __name__ == "__main__":
    main()