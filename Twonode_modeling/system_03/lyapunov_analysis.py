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

save_str = datetime.datetime(2025, 1, 15)
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

# time vector
params['dt'] = 0.01 #x10ms
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
iPert = np.array([0., 0.15, 0.25, 0.35, 0.5])[::-1]

k_min = -1; k_max = 1; k_steps = 10
eval_k = k_min + np.arange(0, k_steps + 1) / k_steps * (k_max - k_min)

eval_sigma_e = np.array([2.5, 7.5]); sige_steps = len(eval_sigma_e)
eval_sigma_i = np.array([4.4, 16.5]); sigi_steps = len(eval_sigma_i) 

sigVar = np.array([0.1, 1, 2])

for nV, _sigvar in enumerate(sigVar):
    for nI, I in enumerate(iPert):
        params['i_e1'] = I / params['gamma'] + params['i_e2']
        print(f"Evaluating for I = {params['i_e1']:.3f}mV")

        for k_in, K in enumerate(eval_k):
            tic = time.time()
            params['K'] = K
            file_name = f"lyap_K_{params['K']:.2f}_ie_{params['i_e1']:.3f}_sigVar_{_sigvar:.2f}_dt_{params['dt']}ms"

            if os.path.isfile(os.path.join(save_dir, file_name + '.pkl')):
                print(f"file found at {os.path.join(save_dir, file_name + '.pkl')} --> loading file...")
                with open(os.path.join(save_dir, file_name + '.pkl'), 'rb') as f:
                    lyap_mats = pickle.load(f)

            else:
                lyap_mats = {
                    'n1_mean': np.zeros((sige_steps, sigi_steps)),
                    'n1_var': np.zeros((sige_steps, sigi_steps))
                }

                for e1, sigma_e in enumerate(eval_sigma_e):
                    # tic = time.time()
                    for i1, sigma_i in enumerate(eval_sigma_i):
                        params['sig_e1'] = [sigma_e] * n_trials
                        params['sig_i1'] = [sigma_i] * n_trials

                        # Creating sigma array with covariance
                        params['sig_e2'] = []
                        params['sig_i2'] = []

                        for q in range(n_trials):
                            np.random.seed(q * 10 + 5)
                            sige2 = abs(np.random.normal(sigma_e, _sigvar, 1))
                            sigi2 = abs(np.random.normal(sigma_i, _sigvar, 1))

                            params['sig_e2'].append(sige2[0])
                            params['sig_i2'].append(sigi2[0])

                        u = np.zeros((n_trials, 4, T))
                        u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

                        _lyap = np.zeros((n_trials, T - t_start - 1))
                        for k in range(n_trials):
                            _diff = np.abs(np.diff(u[k, 0, t_start:])) / (params['dt'] * 10)
                            _lyap[k, :] += np.log(_diff)

                        _meanlyap = np.mean(_lyap, axis=1)

                        lyap_mats['n1_mean'][e1, i1] = np.mean(_meanlyap)
                        lyap_mats['n1_var'][e1, i1] = np.std(_meanlyap)
                    # toc = time.time()
                    # print(f'finishing sigma value --> Execution time: {toc - tic :.3f}s')

                with open(os.path.join(save_dir, file_name + '.pkl'), 'wb') as f:
                    pickle.dump(lyap_mats, f)

            toc = time.time()
            print(f"finishing K = {K:.2f} Execution time: {toc - tic :.3f}s")