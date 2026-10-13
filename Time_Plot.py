import matplotlib.pyplot as plt

# Define the data from the table
mdp_labels = [50, 98, 200, 312, 450]

# Time for iteration values (in seconds)
q_learning = [93.0946, 93.6714, 96.7785, 97.9071, 99.5971]
zap_ql = [137.9392, 245.1411, 845.2697, 2628.3625, 10935.5351]
pre_cond_ql = [108.0419, 120.4623, 277.9409, 441.4880, 685.1778]
zap_od_ql =[142.8822, 161.0551, 252.3984, 429.277, 780.1779]
sql = [87.9067, 89.5970, 90.8751, 94.9, 95.5466]
nesa_ql =[96.4772, 99.6138, 102.036, 103.5231, 106.1912]

# Plot lines with distinct markers for readability
# Plot lines with distinct colors and markers for readability
plt.plot(mdp_labels, q_learning, marker='o', linewidth=2, label='QL', color='#ff7f0e')
plt.plot(mdp_labels, zap_ql, marker='s', linewidth=2, label='ZQL', color='#2ca02c')
plt.plot(mdp_labels, pre_cond_ql, marker='^', linewidth=2, label='GReQ', color='#1f77b4')
plt.plot(mdp_labels, zap_od_ql, marker='D', linewidth=2, label=r'ZQL $\mathcal{O}(d)$', color='#d62728')
plt.plot(mdp_labels, sql, marker='v', linewidth=2, label='SQL', color='#9467bd')
plt.plot(mdp_labels, nesa_ql, marker='P', linewidth=2, label='NeSA QL', color='#8c564b')

# Customize labels and styling
plt.title('Time per Iteration across Different MDP Size', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('MDP Size', fontsize=12, labelpad=10)
plt.ylabel('Time for Iteration (seconds)', fontsize=12, labelpad=10)

# Add grid lines for easier value tracking
plt.grid(True, linestyle='--', alpha=0.6)

# Place the legend
plt.legend(fontsize=11)

# Adjust layout to prevent clipping
plt.tight_layout()
plt.ylim(0,12000)
# Save the plot
plt.savefig('Time_per_iteration.png')