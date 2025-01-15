import os
import sys
import json
import time
import pickle
import datetime
import numba as nb
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
type_results = "lyapunov_dt"

save_str = datetime.datetime(2025, 1, 10)
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
n_trials = 5
t_start = 1000

# setting init values
np.random.seed(0)
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

# setting the variables
iPert = 0.5 / params['gamma'] + params['i_e2']

dt_vect = np.arange(0.1, 0.55, 0.05) / 10; ntsteps = len(dt_vect)

sig_min = 2.5; sig_max = 16.5; sig_steps = 20
eval_sigma = sig_min + np.arange(0, sig_steps) / (sig_steps - 1) * (sig_max - sig_min)

params['K'] = -0.8

results = np.zeros((ntsteps, sig_steps))
file_name = f"lyap_ie_{params['i_e1']:.3f}"

for nt, dt in enumerate(dt_vect):
    params['dt'] = dt #x10ms Comment out this for recreation

    # time vector
    T = params['T'] = int(3000 / (params['dt'] * 10))
    time_ = np.arange(0, T)
    t_end = params['T']

    print(f"Evaluating for dt = {dt:.3f}ms")

    if os.path.isfile(os.path.join(save_dir, file_name + '.pkl')):
        print(f"file found at {os.path.join(save_dir, file_name + '.pkl')} --> loading file...")

        with open(os.path.join(save_dir, file_name + '.pkl'), 'rb') as f:
            lyap_mats = pickle.load(f)

    else:
        for s1, sigma in enumerate(eval_sigma):
            params['sig_e1'] = params['sig_e2'] = params['sig_i1'] = params['sig_i2'] = sigma

            u = np.zeros((n_trials, 4, T))
            u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

            lyapunov_1 = np.zeros((T - t_start - 1))
            for j in range(n_trials):
                diff_1 = np.abs(np.diff(u[j, 0, t_start:])) / (params['dt'] * 10)
                lyapunov_1 += 1 / n_trials * np.log(diff_1)

            results[nt, s1] = np.mean(lyapunov_1)

plt.figure(figsize=(20,20))
plt.pcolormesh(results)
plt.savefig(os.path.join(save_dir, file_name + '.png'), dpi=600, bbox_inches='tight')

with open(os.path.join(save_dir, file_name + '.pkl'), 'wb') as f:
    pickle.dump(results, f)

