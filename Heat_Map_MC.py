import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ==========================================
# 1. PATH AND CORE LAYOUT MANAGEMENT
# ==========================================
BASE_DATA_DIR = "./Data"  # Ensure this points to the folder containing 'Markov_Sampling'

# Human-readable labels for your 10 standard exponent combinations
EXPONENT_LABELS = [
    r"C1: $\alpha_{1.0}, \beta_{0.9}$", r"C2: $\alpha_{1.0}, \beta_{0.8}$",
    r"C3: $\alpha_{1.0}, \beta_{0.7}$", r"C4: $\alpha_{1.0}, \beta_{0.6}$",
    r"C5: $\alpha_{0.9}, \beta_{0.8}$", r"C6: $\alpha_{0.9}, \beta_{0.7}$",
    r"C7: $\alpha_{0.9}, \beta_{0.6}$", r"C8: $\alpha_{0.8}, \beta_{0.7}$",
    r"C9: $\alpha_{0.8}, \beta_{0.6}$", r"C10: $\alpha_{0.7}, \beta_{0.6}$"
]
COLUMN_LABEL = ["Markov Trajectories"]

def get_final_metric_value(alg_prefix, alg_name, mdp, gamma, trial_num, metric='value_norm'):
    """
    Constructs the exact string path matching your Markov filename layout:
    Data/Markov_Sampling/{alg_prefix}_{mdp}_Random/{gamma}/{alg_prefix}_Trial_{num}_results.pkl
    """
    alg_folder = f"{alg_prefix}_{mdp}_Markov"
    clean_gamma_val = gamma.replace('_', '.')
    
    # Matches your Markov naming convention: e.g., Pre-Cond_Q-Learning_0.999_Trial_1_results.pkl
    file_name = f"{alg_name}_{clean_gamma_val}_Trial_{trial_num}_results.pkl"
    file_path = os.path.join(BASE_DATA_DIR, "Markov_Sampling", alg_folder, gamma, file_name)
    
    if not os.path.exists(file_path):
        return None  # Catches diverged or ungenerated runs cleanly

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
            # Safe float conversion to handle explicit 2D indexing
            values = np.array(data['metrics'][metric], dtype=np.float64)
            
            # Access the final column (checkpoint 1000) across all 5 seeds
            final_step_idx = values.shape[1] - 1
            final_step_values = values[:, final_step_idx]
            
            return np.median(final_step_values)
    except Exception:
        return None

# ==========================================
# 2. EVALUATION COMPILATION ENGINE
# ==========================================
def compile_markov_log_ratio_matrix(mdp, gamma, metric='value_norm'):
    """Loops through all 10 sequential trials and maps them to a 10x1 vertical strip."""
    grid = np.zeros((10, 1))
    
    # Establish strict numerical clipping bounds to completely prevent overflow/underflow anomalies
    LOWER_UNDERFLOW_LIMIT = 1e-10
    UPPER_OVERFLOW_LIMIT  = 1e10
    
    # Markov tracking loops strictly through the 10 exponent trials
    for trial in range(1, 11):
        row_idx = trial - 1
        
        # Pull matching values using your exact directory prefixes
        ve_pre = get_final_metric_value("Pre_Cond_QL", "Pre-Cond_Q-Learning_(Trajectory)", mdp, gamma, trial, metric)
        ve_zap = get_final_metric_value("Zap_QL", "ZAP_Q-Learning_(Trajectory)", mdp, gamma, trial, metric)

        
        # 1. Handle explicit file-not-found or tracking script divergence crashes
        if ve_pre is None and ve_zap is None:
            grid[row_idx, 0] = 0.0
            continue
        elif ve_zap is None:  
            grid[row_idx, 0] = -3.0  # Zap missing/failed: Pre-conditioned Win
            continue
        elif ve_pre is None:  
            grid[row_idx, 0] = 3.0   # Pre-conditioned missing/failed: Zap Win
            continue
            
        # 2. Check for mathematical overflow/underflow values inside the parsed scalars
        is_pre_invalid = np.isnan(ve_pre) or np.isinf(ve_pre) or (ve_pre > UPPER_OVERFLOW_LIMIT)
        is_zap_invalid = np.isnan(ve_zap) or np.isinf(ve_zap) or (ve_zap > UPPER_OVERFLOW_LIMIT)
        
        if is_pre_invalid and is_zap_invalid:
            grid[row_idx, 0] = 0.0  # Both broke down completely
        elif is_zap_invalid:
            grid[row_idx, 0] = -3.0  # Zap blew up to infinity, Pre-conditioned stabilized
        elif is_pre_invalid:
            grid[row_idx, 0] = 3.0   # Pre-conditioned blew up to infinity
        else:
            # 3. Safe, bounded calculation for valid numeric ranges
            safe_pre = max(ve_pre, LOWER_UNDERFLOW_LIMIT)
            safe_zap = max(ve_zap, LOWER_UNDERFLOW_LIMIT)
            
            # Calculate the direct log-ratio step safely
            log_ratio = np.log10(safe_pre / safe_zap)
            
            # Clip the final output score so it stays within your [-3.0, 3.0] visual color spectrum
            grid[row_idx, 0] = np.clip(log_ratio, -3.0, 3.0)
            
    return pd.DataFrame(grid, index=EXPONENT_LABELS, columns=COLUMN_LABEL)

# ==========================================
# 3. HEATMAP PLOTTING INTERFACE
# ==========================================
def render_thesis_heatmap(df_data, mdp, gamma, save_path="./plots"):
    os.makedirs(save_path, exist_ok=True)
    
    # Narrower aspect ratio (5.5, 8) optimized for a single column vertical strip
    plt.figure(figsize=(5.5, 8))
    
    # Symmetrical diverging palette: Dark Green (Win) -> Clear White (Parity) -> Dark Red (Loss)
    cmap_diverging = mcolors.LinearSegmentedColormap.from_list("GWR", ["#238b45", "#ffffff", "#d73027"])
    
    # Enforce balanced visual alignment around zero midpoint
    color_bound = max(abs(df_data.values.min()), abs(df_data.values.max()), 1.0)
    
    sns.heatmap(
        df_data, cmap=cmap_diverging, center=0.0, vmin=-color_bound, vmax=color_bound,
        annot=True, fmt=".2f", linewidths=1.0, linecolor='#e2e2e2',
        cbar_kws={'label': r'Logarithmic Precision Index $\log_{10}(\text{Value Error}_{\text{Pre}} / \text{Value Error}_{\text{Zap}})$'}
    )
    
    clean_gamma = gamma.replace('_', '.')
    plt.title(f"Asymptotic Log-Ratio Stability Mapping\n(Asynchronous Markov Sampling)\nEnvironment: {mdp} | $\gamma = {clean_gamma}$", 
              fontsize=11, pad=15, weight='bold')
    plt.ylabel("Two-Timescale Exponent Groupings", fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)
    
    plt.tight_layout()
    output_filename = f"markov_heatmap_{mdp}_{gamma}.png"
    plt.savefig(os.path.join(save_path, output_filename), dpi=300)
    print(f"Success! Markov strip heatmap saved to: {os.path.join(save_path, output_filename)}")
    plt.show()

# ==========================================
# 4. EXECUTION CONTROLLER
# ==========================================
if __name__ == "__main__":
    # Choose your targeted slice to plot
    chosen_mdp = "Graph"       # Options matching your directory: 'Garnet', 'Graph', 'Grid'
    chosen_gamma = "0_9"       # Options matching your directory: '0_9', '0_99', '0_999'
    
    print(f"Compiling Markov trajectory sampling strip for {chosen_mdp} at gamma={chosen_gamma}...")
    
    # 1. Execute pipeline parse
    df_matrix = compile_markov_log_ratio_matrix(mdp=chosen_mdp, gamma=chosen_gamma, metric='value_norm')
    
    # 2. Render column strip
    render_thesis_heatmap(df_matrix, mdp=chosen_mdp, gamma=chosen_gamma)