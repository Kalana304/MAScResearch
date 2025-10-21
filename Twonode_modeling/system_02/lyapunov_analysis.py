import os
import sys
import json
import time
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

sys.path.insert(1, '../utils')

from utils import *
from plotUtils import *
from mUtils import *

plt.rcParams.update({'font.size': 14})

# Setting save dir paths
root_dir = "./results"
type_results = "lyapunov"

save_str = datetime.datetime(2025, 1, 13)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"

save_dir = os.path.join(root_dir, type_results, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)

# read parameters from a json file
with open('../params.json', 'r+') as f:
    params = json.load(f)

# setting grid parameters
minval = -5 / params['gamma']
maxval = 5 / params['gamma']
grid_res = 100
n_trials = 10

params['dt'] = 0.005 #x10ms Comment out this for recreation

# time vector
T = params['T'] = int(params['T'] / (params['dt'] * 10))
time_ = np.arange(0, T)

# setting init values
np.random.seed(0)
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

t_start = int(1000 / (params['dt'] * 10))
t_end = params['T']

# setting variables
iPert = np.array([0.  , 0.15, 0.25, 0.35, 0.5])[::-1]

k_min = -1; k_max = 1; k_steps = 10
eval_k = k_min + np.arange(0, k_steps + 1) / k_steps * (k_max - k_min)

sige_min = 2.5; sige_max = 16.5; sige_steps = 20
sigi_min = 2.5; sigi_max = 16.5; sigi_steps = 20

eval_sigma_e = sige_min + np.arange(0, sige_steps) / (sige_steps - 1) * (sige_max - sige_min)
eval_sigma_i = sigi_min + np.arange(0, sigi_steps) / (sigi_steps - 1) * (sigi_max - sigi_min)

for nI, I in enumerate(iPert):
    # setting the constant value
    params['i_e1'] = I / params['gamma'] + params['i_e2']
    print(f"Evaluating for I = {params['i_e1']:.3f}")

    for k_in, K in enumerate(eval_k):
        tic = time.time()
        params['K'] = K
        file_name = f"lyap_K_{params['K']:.2f}_ie_{params['i_e1']:.3f}"

        if os.path.isfile(os.path.join(save_dir, file_name + '.pkl')):
            print(f"file found at {os.path.join(save_dir, file_name + '.pkl')} --> loading file...")
            with open(os.path.join(save_dir, file_name + '.pkl'), 'rb') as f:
                lyap_mats = pickle.load(f)

        else:
            lyap_mats = {
                'n1_mean': np.zeros((sige_steps, sigi_steps)),
                'n2_mean': np.zeros((sige_steps, sigi_steps))
            }

            for e1, sigma_e in enumerate(eval_sigma_e):
                params['sig_e1'] = params['sig_e2'] = sigma_e
                # tic = time.time()
                for i1, sigma_i in enumerate(eval_sigma_i):
                    params['sig_i1'] = params['sig_i2'] = sigma_i

                    u = np.zeros((n_trials, 4, T))
                    u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

                    lyapunov_1 = np.zeros((T - t_start - 1))
                    for j in range(n_trials):
                        diff_1 = np.abs(np.diff(u[j, 0, t_start:])) / (params['dt'] * 10)
                        lyapunov_1 += 1 / n_trials * np.log(diff_1)

                    lyapunov_2 = np.zeros((T - t_start - 1))
                    for j in range(n_trials):
                        diff_2 = np.abs(np.diff(u[j, 2, t_start:])) / (params['dt'] * 10)
                        lyapunov_2 += 1 / n_trials * np.log(diff_2)

                    lyap_mats['n1_mean'][e1, i1] = np.mean(lyapunov_1)
                    lyap_mats['n2_mean'][e1, i1] = np.mean(lyapunov_2)
                # toc = time.time()
                # print(f'finishing sigma value --> Execution time: {toc - tic :.3f}s')

            with open(os.path.join(save_dir, file_name + '.pkl'), 'wb') as f:
                pickle.dump(lyap_mats, f)

        toc = time.time()
        print(f"finishing K = {K:.2f} Execution time: {toc - tic :.3f}s")
