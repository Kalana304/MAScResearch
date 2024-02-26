import os
import json
import time
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from utils import *
from plot_utils import *
from measure_utils import *

plt.rcParams.update({'font.size': 14})


# Setting save dir paths
root_dir = "./results"
type_results = "lyapunov"
scene_dir = "scene03v1"

save_str = datetime.datetime(2024, 2, 13)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"

save_dir = os.path.join(root_dir, type_results, scene_dir, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)

# read parameters from a json file
with open('./params.json', 'r+') as f:
    params = json.load(f)

# setting grid parameters
minval = -5 / params['gamma']
maxval = 5 / params['gamma']
grid_res = 1000
n_trials = 10

# time vector
T = params['T']
time_ = np.arange(0, T)

# Initialization points
random.seed(0)

x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

t_start = 1000
t_end = params['T']

# setting the constant value
params['i_e1'] = 0.25 / params['gamma']

# setting the variables
k_min = -1; k_max = 1; k_steps = 10; delta_k = (k_max - k_min) / k_steps
sige_min = 2.5; sige_max = 17.5; sige_steps = 10; delta_sige = (sige_max - sige_min) / sige_steps
sigi_min = 2.5; sigi_max = 17.5; sigi_steps = 20

eval_sigma_i1 = sigi_min + np.arange(0, sigi_steps) / (sigi_steps - 1) * (sigi_max - sigi_min)
eval_sigma_i2 = sigi_min + np.arange(0, sigi_steps) / (sigi_steps - 1) * (sigi_max - sigi_min)

# lyapunov exponenet response maps sigma_e vs. sigma_i

if_plot_lyap = False
if_save_lyap = False
mask_thresh = 0.02

for k in np.arange(k_min, k_max + delta_k, delta_k):
    params['K'] = k

    for sige in np.arange(sige_min, sige_max + delta_sige, delta_sige):
        params['sig_e1'] = params['sig_e2'] = sige 
        file_name = f"lyap_K_{params['K']:.2f}_ie_{params['i_e1']:.3f}_sige_{sige:.3f}"

        if os.path.isfile(os.path.join(save_dir, file_name + '.pkl')):
            print(f"file found at {os.path.join(save_dir, file_name + '.pkl')} --> loading file...")
            with open(os.path.join(save_dir, file_name + '.pkl'), 'rb') as f:
                lyap_mats = pickle.load(f)
        
        else:
            lyap_mats = {
                'n1_mean': np.zeros((sigi_steps, sigi_steps)),
                'n2_mean': np.zeros((sigi_steps, sigi_steps)),
                'n1_max': np.zeros((sigi_steps, sigi_steps)),
                'n2_max': np.zeros((sigi_steps, sigi_steps))
            }
            for ns1, sigi_1 in enumerate(eval_sigma_i1):
                params['sig_i1'] = sigi_1 
                tic = time.time()

                for ns2, sigi_2 in enumerate(eval_sigma_i2):
                    params['sig_i2'] = sigi_2

                    u = np.zeros((n_trials, 4, T))
                    u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

                    lyapunov_1 = np.zeros((T - t_start - 1))
                    for k in range(n_trials):
                        diff_1 = np.abs(np.diff(u[k, 0, t_start:]))
                        lyapunov_1 += 1 / n_trials * np.log(diff_1)

                    lyapunov_2 = np.zeros((T - t_start - 1))
                    for k in range(n_trials):
                        diff_2 = np.abs(np.diff(u[k, 2, t_start:]))
                        lyapunov_2 += 1 / n_trials * np.log(diff_2)

                    lyap_mats['n1_mean'][ns1, ns2] = np.mean(lyapunov_1)
                    lyap_mats['n2_mean'][ns1, ns2] = np.mean(lyapunov_2)
                    lyap_mats['n1_max'][ns1, ns2] = lyapunov_1.max()
                    lyap_mats['n2_max'][ns1, ns2] = lyapunov_2.max()

                toc = time.time()
                print(f'finishing sigma value --> Execution time: {toc - tic :.3f}s')

            with open(os.path.join(save_dir, file_name + '.pkl'), 'wb') as f:
                pickle.dump(lyap_mats, f)


