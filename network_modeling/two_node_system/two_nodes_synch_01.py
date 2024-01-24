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

from scipy.stats import norm
from sklearn.feature_selection import mutual_info_regression as MI

from utils import *
from plot_utils import *
from measure_utils import *

plt.rcParams.update({'font.size': 16})

# read parameters from a json file
with open('./params.json', 'r+') as f:
    params = json.load(f)

root_dir = 'results/synchron'
scene_sub_dir = 'scene01'

save_str = datetime.datetime(2024, 1, 23)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"

save_dir = os.path.join(root_dir, scene_sub_dir, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)

# setting grid parameters
minval = -5 / params['gamma']
maxval = 5 / params['gamma']
grid_res = 100
n_trials = 10

# time vector
T = params['T']
time_ = np.arange(0, T)

t_start = 1000
t_end = params['T']

# setting init values
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

# setting the constant value
params['i_e1'] = 0.0 / params['gamma']

# setting variables
sig_min = 2.5; sig_max = 17.5; sig_steps = 20
eval_sigma = sig_min + np.arange(0, sig_steps) / sig_steps * (sig_max - sig_min)

k_min = -1; k_max = 1; k_steps = 10
eval_K = k_min + np.arange(0, k_steps) / k_steps * (k_max - k_min)

thresh_aval = 1.4

for k_ind, K in enumerate(eval_K):
    print(f'Running for K = {K:.3f} ----> ', end='')
    params['K'] = K

    CVStability = np.zeros((sig_steps, n_trials))

    AvalanchSync = np.zeros((sig_steps, n_trials))
    LocalmaxSync = np.zeros((sig_steps, n_trials))
    KuramotoOrder = np.zeros((sig_steps, n_trials))

    GlobalSync = np.zeros((sig_steps, n_trials))
    SVDSync = np.zeros((sig_steps, n_trials))
    MI_phase = np.zeros((sig_steps, n_trials))
    MI_amp = np.zeros((sig_steps, n_trials))

    tic = time.time()
    for si, sigma_ in enumerate(eval_sigma):
        # print(f"Running evaluation {si} for sigma = {sigma_:.3f}")
        params['sig_e1'] = params['sig_e2'] = params['sig_i1'] = params['sig_i2'] = sigma_

        u = np.zeros((n_trials, 4, T))
        u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

        # Avalanch synch calculation
        ASynch = AvalanchSynch(time_series=u, thresh1=thresh_aval, thresh2=thresh_aval, tstart=t_start, tend=t_end, if_avg=False)
        AvalanchSync[si, :]= ASynch.calc_synchronization()

        # Local maxima synch calculation
        LSynch = LocalMaxSynch(time_series=u, nSamples=10, tstart=t_start, tend=t_end)
        LocalmaxSync[si, :] = LSynch.calc_synchronization()

        # CV value calculation
        CVStability[si, :] = CoeffVariance(time_series=u, tstart=t_start, tend=t_end)

        # Kuramoto Order parameter
        KuramotoOrder[si, :] = kuramoto_order_calc(time_series=u, tstart=t_start, tend=t_end, window_size=200, step_window=50)

        # Spectral synch calculation
        SSync = SpectrumSynch(time_series=u, tstart=t_start, tend=t_end, fmin=35, fmax=50, fs=1000, window_size=200, step_window=50)
        glob_trial, svd_trial, mi_phase_trial, mi_amp_trial = SSync.calc_synchronization()

        GlobalSync[si, :] = glob_trial
        SVDSync[si, :] = svd_trial
        MI_phase[si, :] = mi_phase_trial
        MI_amp[si, :] = mi_amp_trial

    toc = time.time()
    print(f'ETA = {toc - tic:.2f}s\n')

    synchron_data = {
                    'Avalanch': AvalanchSync,
                    'Localmax': LocalmaxSync,
                    'Kuramoto': KuramotoOrder,
                    'GlobalSync': GlobalSync,
                    'SVD': SVDSync,
                    'MI_phase': MI_phase,
                    'MI_amp': MI_amp,
                    'CV': CVStability

    }

    file_name = f"results_K_{params['K']:.2f}_I_{params['i_e1']:.3f}mV.pkl"

    with open(os.path.join(save_dir, file_name), 'wb') as _file:
        pickle.dump(synchron_data, _file)