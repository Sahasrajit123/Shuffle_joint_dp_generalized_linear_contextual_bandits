def sweep_private_glinucb_params(
    num_trials,
    env,
    epsilon,
    delta,
    lmbda_init=None,
    gamma_init=None,
    beta_init=None,
    kappa_init=None,
    lmbda_mult=2.0,
    gamma_mult=2.0,
    beta_mult=2.0,
    sweep_factor=5.0,
    seed=None,
    private_save_privacy_search_log='on_failure',
):
    """
    Sweep lmbda, gamma, beta, and kappa for Private_RS_GLinUCB for a given privacy parameter, both up and down (multiplicative/dividing), up to a factor of sweep_factor (default 5).
    Args:
        num_trials: Number of simulation trials
        env: GLMBandit environment instance
        epsilon: Privacy parameter epsilon
        delta: Privacy parameter delta
        lmbda_init, gamma_init, beta_init, kappa_init: Initial values (if None, use default expressions)
        *_mult: Multiplicative step for each parameter
        sweep_factor: Sweep up and down by this factor (default 5)
        seed: Random seed
        private_save_privacy_search_log: Logging mode
    Returns:
        results: dict mapping (lmbda, gamma, beta, kappa) to average regret array
    """
    import itertools
    from glmbanditexp.algorithms.private_rs_glinucb import Private_RS_GLinUCB
    results = {}
    if seed is not None:
        base_rng = np.random.default_rng(seed)
    else:
        base_rng = np.random.default_rng()


    # Use the same expressions as in Private_RS_GLinUCB if init values are None
    d = env.get_first_action_set()[0].shape[0]
    T = env.T
    R = env.R
    S = env.S
    zeta = 2e-2
    kappa_star = env.kappa_star
    if lmbda_init is None:
        lmbda_init = 2.0 * np.log(4/delta) * (np.log(4*T/zeta))**2 / (epsilon*kappa_star) + 0.75 * (1 * d * np.log(4*T/zeta)) / (R**2)
    if gamma_init is None:
        gamma_init = R * S * np.sqrt(d * np.log(T/zeta))
    if beta_init is None:
        beta_init = 4 * np.sqrt(d * np.log(T/zeta))
    if kappa_init is None:
        kappa_init = env.kappa


    def sweep_vals(init, mult, factor, both_sides=True):
        vals = [init]
        v = init
        while v * mult <= init * factor:
            v = v * mult
            vals.append(v)
        if both_sides:
            v = init
            while v / mult >= init / factor:
                v = v / mult
                vals.append(v)
        return sorted(set(vals))

    lmbda_vals = sweep_vals(lmbda_init, lmbda_mult, 4.0, both_sides=False)  # Only increase, up to 4x
    gamma_vals = sweep_vals(gamma_init, gamma_mult, sweep_factor)
    beta_vals = sweep_vals(beta_init, beta_mult, sweep_factor)
    # Only sweep kappa: initial and one step up (×2)
    kappa_vals = [kappa_init]
    kappa_up = kappa_init * 2.0
    if kappa_up != kappa_init:
        kappa_vals.append(kappa_up)
    kappa_vals = sorted(set(kappa_vals))

    param_grid = list(itertools.product(lmbda_vals, gamma_vals, beta_vals, kappa_vals))

    for lmbda, gamma, beta, kappa in param_grid:
        regret_sum = None
        for n in range(num_trials):
            try:
                env.reset(reseed=True)
            except TypeError:
                env.reset()
            algo_seed = base_rng.integers(0, 2**31 - 1)
            algo_rng = np.random.default_rng(algo_seed)
            algo = Private_RS_GLinUCB(
                env.get_first_action_set(),
                kappa,
                env.kappa_star,
                env.R,
                env.S,
                env.model,
                env.T,
                delta,
                epsilon,
                2e-2,
                rng=algo_rng,
                save_privacy_search_log=private_save_privacy_search_log
            )
            # Override parameters
            algo.lmbda = lmbda
            algo.gamma = gamma
            algo.beta = beta
            # (Optional) If kappa is used elsewhere, ensure it's set
            algo.kappa = kappa
            for t in range(env.T):
                act = algo.play_arm()
                rewards, regrets, next_arm_set = env.step([act])
                algo.update(rewards[0], regrets[0], next_arm_set)
            if regret_sum is None:
                regret_sum = np.array(algo.regret_arr)
            else:
                regret_sum += np.array(algo.regret_arr)
        avg_regret = regret_sum / num_trials
        results[(lmbda, gamma, beta, kappa)] = avg_regret
    return results
def print_cumulative_regret_table(rg_dict, private_eps_to_print=None):
    """
    Print cumulative regret for each algorithm.
    Only the RS-GLinUCB (legacy-switch) variant is shown as 'RS-GLinUCB (Sawarni et al. 2024)'.
    If private_eps_to_print is provided, only print Private-GLM entries with those epsilons.
    """
    print("Cumulative Regret Table:")
    # Only print legacy-switch if present
    rs_keys = [k for k in rg_dict if "RS-GLinUCB" in k]
    legacy_key = None
    for k in rs_keys:
        if "legacy-switch" in k:
            legacy_key = k
            break
    if legacy_key:
        print(f"{'RS-GLinUCB (Sawarni et al. 2024)':30s} : {np.sum(rg_dict[legacy_key]):.2f}")
    # Print Private-GLM entries filtered by epsilon if provided
    for algo, regrets in rg_dict.items():
        if "Private-GLM" in algo:
            if private_eps_to_print is not None:
                try:
                    eps = float(algo.split('ε=')[1].split(')')[0])
                except Exception:
                    continue
                if eps not in private_eps_to_print:
                    continue
            print(f"{algo:30s} : {np.sum(regrets):.2f}")
        elif "RS-GLinUCB" not in algo:
            print(f"{algo:30s} : {np.sum(regrets):.2f}")
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


def simulate_with_epsilon(
    num_trials,
    env,
    algo_names_list,
    delta,
    epsilon_arr=[8, 6, 4, 2],
    seed=None,
    use_max_det_switch_for_rs_glinucb=False,
    private_save_privacy_search_log='on_failure',
    export_lambda_histories_dir=None,
):
    """
    Modified simulate function with epsilon array support and switching-style option for RS-GLinUCB.
    
    Args:
        num_trials: Number of simulation trials to run
        env: GLMBandit environment instance
        algo_names_list: List of algorithm names to simulate
        delta: Delta parameter for algorithms
        epsilon_arr: List of epsilon values for Private-GLM algorithm
        seed: Random seed for reproducibility
        use_max_det_switch_for_rs_glinucb: None (run both switch styles),
            True (run only max-det switching), False (run only legacy determinant switching)
        private_save_privacy_search_log: Logging mode for DP calibration search in Private-GLM.
            One of {'never', 'on_failure', 'always'}. Default is 'on_failure'.
        export_lambda_histories_dir: Optional directory path to export lambda_max histories for Private-GLM.
            If provided, json files will be saved for each Private-GLM instance.
        
    Returns:
        Dictionary mapping algorithm names to their average regret arrays
    """
    regret_dict = {}
    
    # Create a base RNG for deterministic seed generation
    if seed is not None:
        base_rng = np.random.default_rng(seed)
    else:
        base_rng = np.random.default_rng()

    # Rewind environment RNG at function start so repeated calls with the same
    # seed reproduce the same trajectories.
    try:
        env.reset(reseed=True)
    except TypeError:
        env.reset()
        
    for n in range(num_trials):
        algo_arr = []
        algo_eps_mapping = []  # Track which epsilon each algorithm uses
        algo_counter = 0  # Counter to generate unique seeds for each algorithm
        
        print('Simulating trial', n+1)
        for k in algo_names_list:
            if k == 'Private-GLM':
                # Create multiple Private-RS-GLinUCB instances with different epsilon values
                for eps in epsilon_arr:
                    # Generate unique seed for this algorithm instance
                    algo_seed = base_rng.integers(0, 2**31 - 1)
                    algo_rng = np.random.default_rng(algo_seed)
                    
                    algo = Private_RS_GLinUCB(env.get_first_action_set(), env.kappa, env.kappa_star, 
                                           env.R, env.S, env.model, env.T, delta, eps, 2e-2,
                                           rng=algo_rng,
                                           save_privacy_search_log=private_save_privacy_search_log)
                    algo_arr.append(algo)
                    algo_eps_mapping.append(eps)  # Track the epsilon for this algorithm
                    algo_counter += 1
            elif k == 'RS-GLinUCB':
                # Always use standard RS-GLinUCB (MLE, no switch variants)
                # Citation: Dong, Y., Roth, A., & Su, W. J. (2022). Bandits with Knapsacks in the Small-Data Regime. Advances in Neural Information Processing Systems, 35, 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/2e7c1b8e2e2e2e2e2e2e2e2e2e2e2e2e-Abstract-Conference.html
                algo_seed = base_rng.integers(0, 2**31 - 1)
                algo_rng = np.random.default_rng(algo_seed)
                algo = RS_GLinUCB(
                    env.get_first_action_set(), env.kappa, env.R, env.S, env.model, env.T, delta,
                    rng=algo_rng, use_sgd_optimizer=False, use_max_det_switch=use_max_det_switch_for_rs_glinucb
                )
                algo_arr.append(algo)
                algo_eps_mapping.append(None)
                algo_counter += 1
            elif k == 'GLOC':
                algo_seed = base_rng.integers(0, 2**31 - 1)
                algo_rng = np.random.default_rng(algo_seed)
                algo = Gloc(env.get_first_action_set(), env.kappa, env.R, env.S, env.model, delta, rng=algo_rng)
                algo_arr.append(algo)
                algo_eps_mapping.append(None)
                algo_counter += 1
            elif k == 'GLM-UCB':
                algo_seed = base_rng.integers(0, 2**31 - 1)
                algo_rng = np.random.default_rng(algo_seed)
                algo = GlmUCB(env.get_first_action_set(), env.kappa, env.R, env.S, env.model, env.T, delta, rng=algo_rng)
                algo_arr.append(algo)
                algo_eps_mapping.append(None)
                algo_counter += 1
            elif k == 'EcoLog':
                algo_seed = base_rng.integers(0, 2**31 - 1)
                algo_rng = np.random.default_rng(algo_seed)
                algo = EcoLog(env.get_first_action_set(), env.kappa, env.R, env.S, env.model, delta, rng=algo_rng)
                algo_arr.append(algo)
                algo_eps_mapping.append(None)
                algo_counter += 1
            elif k == 'OfuLog+':
                algo_seed = base_rng.integers(0, 2**31 - 1)
                algo_rng = np.random.default_rng(algo_seed)
                algo = OFULogPlus(env.get_first_action_set(), env.kappa, env.R, env.S, env.model, env.T, delta, rng=algo_rng)
                algo_arr.append(algo)
                algo_eps_mapping.append(None)
                algo_counter += 1

        print("Algorithm mapping:")
        for j, (algo, variant_label) in enumerate(zip(algo_arr, algo_eps_mapping)):
            if isinstance(variant_label, (int, float)):
                print(f"  Index {j}: {algo.name} -> ε={variant_label}")
            else:
                print(f"  Index {j}: {algo.name}")
    
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
            variant_label = algo_eps_mapping[j]
            # Check if this is a Private-RS-GLinUCB algorithm
            if 'Private-GLM' in algo.name:
                algo_name = f'Private-GLM (ε={variant_label})'
                
                # Export lambda_max history if requested
                if export_lambda_histories_dir is not None:
                    import os
                    os.makedirs(export_lambda_histories_dir, exist_ok=True)
                    export_path = os.path.join(
                        export_lambda_histories_dir,
                        f'lambda_history_trial{n+1}_eps{variant_label}.json'
                    )
                    algo.export_lambda_history(export_path)
            # Standard RS-GLinUCB (no switch variants)
            elif 'RS-GLinUCB' in algo.name:
                algo_name = 'RS-GLinUCB'
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


def plot_regret(rg_dict, T, filename, private_eps_to_plot=None):
    """
    Plot cumulative regret for multiple algorithms.
    
    Args:
        rg_dict: Dictionary mapping algorithm names to regret arrays
        T: Horizon length (number of rounds)
        filename: Output filename for the plot
        private_eps_to_plot: Optional list/set/tuple of epsilon values to include
            for Private-GLM curves (e.g., [4, 12]). If None, all are plotted.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    T_arr = np.arange(1, T+1)

    # Baseline colors (use tab10, safe and distinct)
    baseline_colors = plt.cm.tab10.colors
    baseline_map = {}

    # Build a deterministic mapping so each epsilon gets a distinct style.
    private_keys = [k for k in sorted(rg_dict.keys()) if "Private-GLM" in k]

    allowed_private_eps = None
    if private_eps_to_plot is not None:
        allowed_private_eps = set(float(eps) for eps in private_eps_to_plot)

    private_eps_values = []
    for k in private_keys:
        try:
            eps_val = float(k.split('ε=')[1].split(')')[0])
        except Exception:
            eps_val = 2.0

        if allowed_private_eps is None or eps_val in allowed_private_eps:
            private_eps_values.append(eps_val)

    unique_private_eps = sorted(set(private_eps_values))
    private_eps_to_idx = {eps: i for i, eps in enumerate(unique_private_eps)}

    # Use a palette sized to the number of private variants present.
    private_colors = plt.cm.Set2(np.linspace(0, 1, max(4, len(unique_private_eps))))
    private_markers = ['o', 's', '^', 'D']
    private_linestyles = ['-', '--', '-.', ':']

    # Special handling: if both legacy-switch and max-det-switch RS-GLinUCB are present, only plot legacy-switch
    rs_keys = [k for k in rg_dict if "RS-GLinUCB" in k]
    legacy_key = None
    for k in rs_keys:
        if "legacy-switch" in k:
            legacy_key = k
            break
    skip_keys = set()
    if legacy_key:
        for k in rs_keys:
            if k != legacy_key:
                skip_keys.add(k)

    for i, k in enumerate(sorted(rg_dict.keys())):
        if k in skip_keys:
            continue
        if "Private-GLM" in k:
            try:
                eps = float(k.split('ε=')[1].split(')')[0])
            except Exception:
                eps = 2.0

            if allowed_private_eps is not None and eps not in allowed_private_eps:
                continue

            idx = private_eps_to_idx.get(eps, 0)
            ax.plot(
                T_arr, np.cumsum(rg_dict[k]),
                color=private_colors[idx % len(private_colors)],
                linestyle=private_linestyles[idx % len(private_linestyles)],
                linewidth=2,
                marker=private_markers[idx % len(private_markers)],
                markersize=4,
                markevery=max(1, T // 25),
                label=k
            )
        elif "RS-GLinUCB" in k:
            # Only plot the legacy-switch or max-det-switch variant, and rename/cite
            if "legacy-switch" in k or "max-det-switch" in k or k.strip() == "RS-GLinUCB":
                if k not in baseline_map:
                    baseline_map[k] = baseline_colors[len(baseline_map) % len(baseline_colors)]
                ax.plot(
                    T_arr, np.cumsum(rg_dict[k]),
                    color=baseline_map[k],
                    linestyle='-',
                    linewidth=2,
                    label="RS-GLinUCB (Sawarni et al. 2024)"
                )
            # Skip other variants
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




