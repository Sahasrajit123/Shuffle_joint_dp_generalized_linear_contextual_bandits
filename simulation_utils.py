"""
Simulation utilities for GLM bandit experiments.

This module provides functions for running simulations and plotting results.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from glmbanditexp.algorithms.ecolog import EcoLog
from glmbanditexp.algorithms.gloc import Gloc
from glmbanditexp.algorithms.glm_ucb import GlmUCB
from glmbanditexp.algorithms.rs_glinucb import RS_GLinUCB
from glmbanditexp.algorithms.private_rs_glinucb import Private_RS_GLinUCB
from glmbanditexp.algorithms.ofulogplus import OFULogPlus


def simulate_with_epsilon(num_trials, env, algo_names_list, delta, epsilon_arr=[8, 6, 4, 2], seed=None):
    """
    Modified simulate function with epsilon array support.
    
    Args:
        num_trials: Number of simulation trials to run
        env: GLMBandit environment instance
        algo_names_list: List of algorithm names to simulate
        delta: Delta parameter for algorithms
        epsilon_arr: List of epsilon values for Private-GLM algorithm
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping algorithm names to their average regret arrays
    """
    regret_dict = {}
    # Create a single rng for all trials
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    for n in range(num_trials):
        algo_arr = []
        algo_eps_mapping = []  # Track which epsilon each algorithm uses
        print('Simulating trial', n+1)
        for k in algo_names_list:
            if k == 'Private-GLM':
                # Create multiple Private-RS-GLinUCB instances with different epsilon values
                for eps in epsilon_arr:
                    algo = Private_RS_GLinUCB(env.get_first_action_set(), env.kappa, env.kappa_star, 
                                           env.R, env.S, env.model, env.T, delta, eps, 2e-2, rng=rng)
                    algo_arr.append(algo)
                    algo_eps_mapping.append(eps)  # Track the epsilon for this algorithm
            elif k == 'RS-GLinUCB':
                algo = RS_GLinUCB(env.get_first_action_set(), env.kappa, env.R, env.S, env.model, env.T, delta, rng=rng)
                algo_arr.append(algo)
                algo_eps_mapping.append(None)  # No epsilon for non-private algorithms
            elif k == 'GLOC':
                algo = Gloc(env.get_first_action_set(), env.kappa, env.R, env.S, env.model, delta, rng=rng)
                algo_arr.append(algo)
                algo_eps_mapping.append(None)
            elif k == 'GLM-UCB':
                algo = GlmUCB(env.get_first_action_set(), env.kappa, env.R, env.S, env.model, env.T, delta, rng=rng)
                algo_arr.append(algo)
                algo_eps_mapping.append(None)
            elif k == 'EcoLog':
                algo = EcoLog(env.get_first_action_set(), env.kappa, env.R, env.S, env.model, delta, rng=rng)
                algo_arr.append(algo)
                algo_eps_mapping.append(None)
            elif k == 'OfuLog+':
                algo = OFULogPlus(env.get_first_action_set(), env.kappa, env.R, env.S, env.model, env.T, delta, rng=rng)
                algo_arr.append(algo)
                algo_eps_mapping.append(None)

        print("Algorithm mapping:")
        for j, (algo, eps) in enumerate(zip(algo_arr, algo_eps_mapping)):
            print(f"  Index {j}: {algo.name} -> ε={eps}")
    
        for t in tqdm(range(env.T)):
            act_arr = []
            for algo in algo_arr:
                act_arr.append(algo.play_arm())
            try:
                rewards, regrets, next_arm_set = env.step(act_arr)
            except:
                raise ValueError("No action  in next time step")

            for j, algo in enumerate(algo_arr):                
                algo.update(rewards[j], regrets[j], next_arm_set)
        
        # Store results for each algorithm with proper naming
        for j, algo in enumerate(algo_arr):
            # Check if this is a Private-RS-GLinUCB algorithm by checking the algorithm name
            if 'Private-GLM' in algo.name:
                # Use the tracked epsilon value
                eps = algo_eps_mapping[j]
                algo_name = f'Private-GLM (ε={eps})'
            else:
                algo_name = algo.name
            
            if algo_name not in regret_dict.keys():
                regret_dict[algo_name] = np.array(algo.regret_arr)
            else:
                regret_dict[algo_name] += np.array(algo.regret_arr)
        env.reset()
    
    for k in regret_dict.keys():
        regret_dict[k] /= num_trials
    
    return regret_dict


def plot_regret(rg_dict, T, filename):
    """
    Plot cumulative regret for multiple algorithms.
    
    Args:
        rg_dict: Dictionary mapping algorithm names to regret arrays
        T: Horizon length (number of rounds)
        filename: Output filename for the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    T_arr = np.arange(1, T+1)

    # Baseline colors (use tab10, safe and distinct)
    baseline_colors = plt.cm.tab10.colors
    baseline_map = {}

    # Use a distinct colormap for Private-GLM (e.g. Set2 or Dark2)
    private_colors = plt.cm.Set2(np.linspace(0, 1, 4))
    private_markers = ['o', 's', '^', 'D']
    private_linestyles = ['-', '--', '-.', ':']

    for i, k in enumerate(sorted(rg_dict.keys())):
        if "Private-GLM" in k:
            try:
                eps = int(k.split('ε=')[1].split(')')[0])
            except Exception:
                eps = 2
            idx = (eps // 2 - 1) % 4
            ax.plot(
                T_arr, np.cumsum(rg_dict[k]),
                color=private_colors[idx],
                linestyle=private_linestyles[idx % len(private_linestyles)],
                linewidth=2,
                marker=private_markers[idx],
                markersize=4,
                markevery=max(1, T // 25),
                label=k
            )
        else:
            if k not in baseline_map:
                baseline_map[k] = baseline_colors[len(baseline_map) % len(baseline_colors)]
            ax.plot(
                T_arr, np.cumsum(rg_dict[k]),
                color=baseline_map[k],
                linestyle='-',
                linewidth=2,
                label=k
            )

    # Single clean legend (no cutoff issues)
    ax.legend(
        fontsize=11, framealpha=0.9,
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0
    )

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('# of Rounds', fontsize=14)
    ax.set_ylabel('Cumulative Regret', fontsize=14)
    ax.set_title('Cumulative Regret vs # of Rounds', fontsize=16)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()




