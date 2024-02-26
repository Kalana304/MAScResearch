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

save_str = datetime.datetime(2024, 2, 23)
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
T = params['T'] = 10000
time_ = np.arange(0, T)

t_start = 1000
t_end = params['T']

# setting init values
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

# setting the constant value
params['i_e1'] = 0.25 / params['gamma']

# setting variables
sig_min = 2.5; sig_max = 17.5; sig_steps = 50
eval_sigma = sig_min + np.arange(0, sig_steps) / sig_steps * (sig_max - sig_min)

k_min = -1; k_max = 1; k_steps = 10
eval_K = k_min + np.arange(0, k_steps) / k_steps * (k_max - k_min)

freq_min = 30
freq_max = 80
avalanch_limit = 1.4; conf_int = round(norm.cdf(-avalanch_limit) * 2, 2)

AvalanchSync = np.zeros((k_steps, sig_steps))
LocalmaxSync = np.zeros((k_steps, sig_steps))
KuramotoOrder = np.zeros((k_steps, sig_steps))

GlobalSync = np.zeros((k_steps, sig_steps))
SVDSync = np.zeros((k_steps, sig_steps))
MI_phase = np.zeros((k_steps, sig_steps))
MI_amp = np.zeros((k_steps, sig_steps))
PLVSync = np.zeros((k_steps, sig_steps))

for k_ind, K in enumerate(eval_K):
    print(f'Running for K = {K:.3f} ----> ', end='')
    params['K'] = K

    tic = time.time()
    for si, sigma_ in enumerate(eval_sigma):
        params['sig_e1'] = params['sig_e2'] = params['sig_i1'] = params['sig_i2'] = sigma_

        u = np.zeros((n_trials, 4, T))
        u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

        # Avalanch synch calculation
        ASynch = AvalanchSynch(time_series=u, thresh1=avalanch_limit, thresh2=avalanch_limit, tstart=t_start, tend=t_end, if_avg=True)
        AvalanchSync[si, :], AvalanchSync_[si] = ASynch.calc_synchronization()

        # Local maxima synch calculation
        LSynch = LocalMaxSynch(time_series=u, nSamples=10, tstart=t_start, tend=t_end)
        LocalmaxSync[si, :] = LSynch.calc_synchronization()

        # CV value calculation
        CVStability[si, :] = CoeffVariance(time_series=u, tstart=t_start, tend=t_end)

        # Lyapunov Exponent calculation (only node 1)
        LyapStability[si, :] = LyapunovStability(time_series=u, tstart=t_start, tend=t_end)

        # Kuramoto Order parameter
        KuramotoOrder[si, :] = kuramoto_order_calc(time_series=u, tstart=t_start, tend=t_end, window_size=200, step_window=50)

        # Spectral synch calculation
        SSync = SpectrumSynch(time_series=u, tstart=t_start, tend=t_end, fmin=freq_min, fmax=freq_max, fs=1000, window_size=200, step_window=50)
        glob_trial, svd_trial, mi_phase_trial, mi_amp_trial, plv_trial = SSync.calc_synchronization()

        GlobalSync[si, :] = glob_trial
        SVDSync[si, :] = svd_trial
        MI_phase[si, :] = mi_phase_trial
        MI_amp[si, :] = mi_amp_trial
        PLVSync[si, :] = plv_trial

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
                    'PLV': PLVSync,
                    'CV': CVStability,
                    'Lyap': LyapStability
    }

    file_name = f"results_K_{params['K']:.2f}_I_{params['i_e1']:.3f}mV.pkl"

    with open(os.path.join(save_dir, file_name), 'wb') as _file:
        pickle.dump(synchron_data, _file)