import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from check_warmup import avg_warmup_count, plot_avg_warmup_count

from environment import GLMBandit

from glmbanditexp.algorithms.ecolog import EcoLog
from glmbanditexp.algorithms.gloc import Gloc
from glmbanditexp.algorithms.glm_ucb import GlmUCB
from glmbanditexp.algorithms.rs_glinucb import RS_GLinUCB
from glmbanditexp.algorithms.private_rs_glinucb import Private_RS_GLinUCB
from glmbanditexp.algorithms.ofulogplus import OFULogPlus

# Import simulation utilities
from simulation_utils import simulate_with_epsilon, plot_regret

num_trials = 10
d=5
K=10
T=5000
kappa=50.0
R=1.0
S=5.0
lmbda=20.0
seed=186329

count = avg_warmup_count(num_trials, d, K, T, kappa, R, S, lmbda, seed)
plot_avg_warmup_count(count, show_flag=True)

config = {"full_norm": True}
config['seed'] = 109832
config['model'] = 'Probit'
config['theta_dim'] = 3
config['num_arms'] = 20
config['theta_norm'] = 2.0
config['horizon_length'] = 5000

env = GLMBandit(config=config)
print('Kappa =', env.kappa)
print('Theta =' , env.theta)

algo_names_list = ['RS-GLinUCB', 'GLM-UCB', 'GLOC', 'Private-GLM']
num_trials = 10

regret_dict = simulate_with_epsilon(num_trials, env, algo_names_list, 0.02, epsilon_arr = [4, 6,8], seed=109832)
with open('./Results/Extras_probit_mod/experiment_probit_uniform_split_S_2_0_non_private.pickle', 'wb') as f:
    pickle.dump(regret_dict, f)
plot_regret(regret_dict, env.T, './Results/Extras_probit_mod/Probit_Regret_S_2_0_non_private.png')

config = {"full_norm": True}
config['seed'] = 109832
config['model'] = 'Probit'
config['theta_dim'] = 3
config['num_arms'] = 20
config['theta_norm'] = 2.5
config['horizon_length'] = 5000

env = GLMBandit(config=config)
print('Kappa =', env.kappa)
print('Theta =' , env.theta)

algo_names_list = ['RS-GLinUCB', 'GLM-UCB', 'GLOC', 'Private-GLM']
num_trials = 10

regret_dict = simulate_with_epsilon(num_trials, env, algo_names_list, 0.02, epsilon_arr = [4, 6,8], seed=109832)
with open('./Results/Extras_probit_mod/experiment_probit_uniform_split_S_2_5_non_private.pickle', 'wb') as f:
    pickle.dump(regret_dict, f)
plot_regret(regret_dict, env.T, './Results/Extras_probit_mod/Probit_Regret_S_2_5_non_private.png')

config = {"full_norm": True}
config['seed'] = 109832
config['model'] = 'Probit'
config['theta_dim'] = 3
config['num_arms'] = 20
config['theta_norm'] = 3.0
config['horizon_length'] = 5000

env = GLMBandit(config=config)
print('Kappa =', env.kappa)
print('Theta =' , env.theta)

algo_names_list = ['RS-GLinUCB', 'GLM-UCB', 'GLOC', 'Private-GLM']
num_trials = 10

regret_dict = simulate_with_epsilon(num_trials, env, algo_names_list, 0.02, epsilon_arr = [4, 6,8], seed=109832)
with open('./Results/Extras_probit_mod/experiment_probit_uniform_split_S_3_0_non_private.pickle', 'wb') as f:
    pickle.dump(regret_dict, f)
plot_regret(regret_dict, env.T, './Results/Extras_probit_mod/Probit_Regret_S_3_0_non_private.png')

