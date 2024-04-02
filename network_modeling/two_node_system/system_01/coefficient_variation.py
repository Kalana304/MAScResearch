import os
import sys
import json
import time
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../utils')

from utils import *
from plotUtils import *
from mUtils import *

plt.rcParams.update({'font.size': 16})

# read parameters from a json file
with open('./params.json', 'r+') as f:
    params = json.load(f)

root_dir = 'results/cvMeasure'

save_str = datetime.datetime(2024, 3, 13)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"

save_dir = os.path.join(root_dir, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)

# setting grid parameters
minval = -5 / params['gamma']
maxval = 5 / params['gamma']
grid_res = 100
n_trials = 10

# time vector
T = params['T'] = 10000
time_ = np.arange(0, T)

t_start = 1000
t_end = params['T']

# setting init values
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

# setting variables
sig_min = 2.5; sig_max = 16.5; sig_steps = 20
eval_sigma = sig_min + np.arange(0, sig_steps) / (sig_steps-1) * (sig_max - sig_min)

k_min = -1; k_max = 1; k_steps = 10
eval_K = k_min + np.arange(0, k_steps) / k_steps * (k_max - k_min)

iPert = [0.  , 0.15, 0.25, 0.35, 0.5] # [0.0, 0.005, 0.05, 0.5]

for i_ind, I in enumerate(iPert):
    params['i_e1'] = I / params['gamma'] + params['i_e2']
    print(f"Running for I = {params['i_e1']:.3f} ----> ", end='')

    tic = time.time()

    CVStability = np.zeros((k_steps, sig_steps))

    for k_ind, K in enumerate(eval_K):
        params['K'] = K

        for si, sigma_ in enumerate(eval_sigma):
            params['sig_e1'] = params['sig_e2'] = params['sig_i1'] = params['sig_i2'] = sigma_

            u = np.zeros((n_trials, 4, T))
            u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

            # CV value calculation
            CVStability_ = CoeffVariance(time_series=u, tstart=t_start, tend=t_end)
            CVStability[k_ind, si] = np.mean(CVStability_)

    toc = time.time()
    print(f'ETA = {toc - tic:.2f}s\n')

    file_name = f"scenario01_I_{params['i_e1']:.3f}mV.pkl"

    with open(os.path.join(save_dir, file_name), 'wb') as _file:
        pickle.dump(CVStability, _file)