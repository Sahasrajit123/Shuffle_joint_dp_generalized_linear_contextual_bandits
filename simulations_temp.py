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
from simulation_utils import plot_regret, simulate_with_epsilon

config = {"full_norm": True}
config['seed'] = 109832
config['model'] = 'Logistic'
config['theta_dim'] = 5
config['num_arms'] = 20
config['theta_norm'] = 3.5
config['horizon_length'] = 15000

env = GLMBandit(config=config)
print('Kappa =', env.kappa)
print('Theta =' , env.theta)

algo_names_list = ['RS-GLinUCB']
num_trials = 10

# Export lambda_max histories to a dedicated directory
regret_dict = simulate_with_epsilon(
    num_trials, env, algo_names_list, 0.02, 
    epsilon_arr=[1, 2, 4, 8], 
    seed=109832,
    export_lambda_histories_dir='./Results/lambda_histories_logistic_uniform_split_non_private'
)
with open('./Results/test_experiment_logistic_uniform_split_non_private.pickle', 'wb') as f:
    pickle.dump(regret_dict, f)
plot_regret(regret_dict, env.T, './Results/Logistic_Regret_non_private.png')
##plot_regret(regret_dict, env.T, './Results/Logistic_Regret.png')

config = {}
config['seed'] = 109832
config['model'] = 'Probit'
config['theta_dim'] = 5
config['num_arms'] = 20
config['theta_norm'] = 3.0
config['horizon_length'] = 5000
config['full_norm'] = True

env = GLMBandit(config=config)
print('Kappa =', env.kappa)
print('Theta =' , env.theta)

algo_names_list = ['RS-GLinUCB']
num_trials = 10

# Export lambda_max histories to a dedicated directory
regret_dict = simulate_with_epsilon(
    num_trials, env, algo_names_list, 0.02, 
    epsilon_arr=[1, 2, 4, 8], 
    seed=10345,
    export_lambda_histories_dir='./Results/lambda_histories_probit_uniform_split_non_private'
)
with open('./Results/experiment_probit_uniform_split_non_private.pickle', 'wb') as f:
    pickle.dump(regret_dict, f)
plot_regret(regret_dict, env.T, './Results/Probit_Regret_non_private.png')