import pickle
import numpy as np
from environment import GLMBandit

# S values to extract (excluding S=2.5)
s_values = [3.0, 3.5, 4.0]

# Dictionary to store results
results = {}
kappa_values = {}

# Calculate kappa for each S value
for s in s_values:
    config = {"full_norm": True}
    config['seed'] = 109832
    config['model'] = 'Logistic'
    config['theta_dim'] = 3
    config['num_arms'] = 20
    config['theta_norm'] = s
    config['horizon_length'] = 10000
    
    env = GLMBandit(config=config)
    kappa_values[s] = env.kappa

# Load regret data
for s in s_values:
    # Load the pickle file (convert dot to underscore for filename)
    s_str = str(s).replace('.', '_')
    filename = f'Results/Extras/experiment_logistic_uniform_split_S_{s_str}.pickle'
    with open(filename, 'rb') as f:
        regret_dict = pickle.load(f)
    
    # Calculate cumulative regret (sum of all regrets)
    cumulative_regrets = {}
    for algo_name, regret_arr in regret_dict.items():
        # Sum the regret array (works with both numpy arrays and lists)
        if hasattr(regret_arr, 'tolist'):
            cumulative_regret = sum(regret_arr.tolist())
        else:
            cumulative_regret = sum(regret_arr)
        cumulative_regrets[algo_name] = cumulative_regret
    
    results[s] = cumulative_regrets

# Create output file
output_file = 'regret_output_table_logistic.txt'
with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CUMULATIVE REGRET ACROSS DIFFERENT ALGORITHMS (LOGISTIC)\n")
    f.write("=" * 80 + "\n\n")
    
    for s in s_values:
        kappa = kappa_values.get(s, 'N/A')
        f.write("┌" + "─" * 78 + "┐\n")
        if kappa != 'N/A':
            f.write(f"│ S = {s:<10} Kappa = {kappa:<10.2f} {'':<54} │\n")
        else:
            f.write(f"│ S = {s:<72} │\n")
        f.write("├" + "─" * 40 + "┬" + "─" * 37 + "┤\n")
        f.write(f"│ {'Algorithm':<38} │ {'Cumulative Regret':<35} │\n")
        f.write("├" + "─" * 40 + "┼" + "─" * 37 + "┤\n")
        
        # Sort by algorithm name for consistent display
        sorted_items = sorted(results[s].items())
        for algo_name, cum_regret in sorted_items:
            f.write(f"│ {algo_name:<38} │ {cum_regret:>35.2f} │\n")
        
        f.write("└" + "─" * 40 + "┴" + "─" * 37 + "┘\n")
        f.write("\n")

# Also print to console
print("\n" + "=" * 80)
print("CUMULATIVE REGRET ACROSS DIFFERENT ALGORITHMS (LOGISTIC)")
print("=" * 80 + "\n")

for s in s_values:
    kappa = kappa_values.get(s, 'N/A')
    print("┌" + "─" * 78 + "┐")
    if kappa != 'N/A':
        print(f"│ S = {s:<10} Kappa = {kappa:<10.2f} {'':<54} │")
    else:
        print(f"│ S = {s:<72} │")
    print("├" + "─" * 40 + "┬" + "─" * 37 + "┤")
    print(f"│ {'Algorithm':<38} │ {'Cumulative Regret':<35} │")
    print("├" + "─" * 40 + "┼" + "─" * 37 + "┤")
    
    # Sort by algorithm name for consistent display
    sorted_items = sorted(results[s].items())
    for algo_name, cum_regret in sorted_items:
        print(f"│ {algo_name:<38} │ {cum_regret:>35.2f} │")
    
    print("└" + "─" * 40 + "┴" + "─" * 37 + "┘")
    print()

print(f"\nResults saved to: {output_file}")

