import os
import sys
import json
import time
import pickle
import datetime
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

sys.path.insert(1, '../utils')

from utils import *
from plotUtils import *
from mUtils import *

plt.rcParams.update({'font.size': 18})

# read parameters from a json file
with open('../params.json', 'r+') as f:
    params = json.load(f)

root_dir = 'results/synchron'

save_str = datetime.datetime(2024, 4, 23)
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

window_size = 250
overlap_size = 50

# setting init values
np.random.seed(0)
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

# setting the variables
iPert = np.array([0.  , 0.15, 0.25, 0.35, 0.5])[::-1]

sig_min = 2.5; sig_max = 16.5; sig_steps = 20
eval_sigma = sig_min + np.arange(0, sig_steps) / (sig_steps - 1) * (sig_max - sig_min)

k_min = -1; k_max = 1; k_steps = 10
eval_k = k_min + np.arange(0, k_steps) / k_steps * (k_max - k_min)

freq_min = 0.5
freq_max = 30
avalanch_limit = 1.65; conf_int = round(norm.cdf(-avalanch_limit) * 2, 2)

for nI, I in enumerate(iPert):
    # setting the constant value
    params['i_e1'] = I / params['gamma'] + params['i_e2']
    print(f"Evaluating for I = {I:.3f}")
        
    for k_ind, K in enumerate(eval_k):
        print(f'Running for K = {K:.3f} ----> ', end='')
        params['K'] = K

        AvalanchSync = np.zeros((sig_steps, n_trials))
        AvalanchEventDist = np.zeros((sig_steps, n_trials))
        KuramotoOrder = np.zeros((sig_steps, n_trials))
        nlcor_ = np.zeros((sig_steps, n_trials))
        distCorr = np.zeros((sig_steps, n_trials))

        GlobalSync = np.zeros((sig_steps, n_trials))
        SVDSync = np.zeros((sig_steps, n_trials))
        MI_phase = np.zeros((sig_steps, n_trials))
        MI_amp = np.zeros((sig_steps, n_trials))
        PLVSync = np.zeros((sig_steps, n_trials))

        file_name = f"results_K_{params['K']:.2f}_I_{params['i_e1']:.3f}mV.pkl"

        if os.path.isfile(os.path.join(save_dir, file_name)):
            print(f"file found -- skipping processing")
            continue

        tic = time.time()
        for si, sigma_ in enumerate(eval_sigma):
            params['sig_e1'] = params['sig_e2'] = params['sig_i1'] = params['sig_i2'] = sigma_

            u = np.zeros((n_trials, 4, T))
            u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

            # Avalanch synch calculation
            ASynch = AvalanchSynch(time_series=u, thresh1=avalanch_limit, thresh2=avalanch_limit, tstart=t_start, tend=t_end)
            AvalanchSync[si, :], AvalanchEventDist[si, :] = ASynch.run()

            # Kuramoto Order parameter
            KuramotoOrder[si, :] = kuramoto_order_calc(time_series=u, tstart=t_start, tend=t_end, window_size=window_size, step_window=overlap_size)

            # Spectral synch calculation
            SSync = SpectrumSynch(time_series=u, tstart=t_start, tend=t_end, fmin=freq_min, fmax=freq_max, fs=1000, window_size=window_size, step_window=overlap_size)
            glob_trial, svd_trial, mi_phase_trial, mi_amp_trial, plv_trial = SSync.calc_synchronization()

            GlobalSync[si, :] = glob_trial
            SVDSync[si, :] = svd_trial
            MI_phase[si, :] = mi_phase_trial
            MI_amp[si, :] = mi_amp_trial
            PLVSync[si, :] = plv_trial

            CORR = DynamicCorr(time_series=u, tstart=t_start, tend=t_end, window_size=window_size, step_window=overlap_size)
            distance_corr, nonlin_corr = CORR.run()

            distCorr[si, :] = distance_corr
            nlcor_[si, :] = nonlin_corr

        toc = time.time()
        print(f'ETA = {toc - tic:.2f}s\n')

        synchron_data = {
                        'Avalanch': AvalanchSync,
                        'AvalanchDist': AvalanchEventDist,
                        'Kuramoto': KuramotoOrder,
                        'GlobalSync': GlobalSync,
                        'SVD': SVDSync,
                        'MI_phase': MI_phase,
                        'MI_amp': MI_amp,
                        'PLV': PLVSync,
                        'DistCor': distCorr,
                        'nlCor': nlcor_

        }

        with open(os.path.join(save_dir, file_name), 'wb') as _file:
            pickle.dump(synchron_data, _file)