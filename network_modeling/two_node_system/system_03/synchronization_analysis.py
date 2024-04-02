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

plt.rcParams.update({'font.size': 14})

# read parameters from a json file
with open('../params.json', 'r+') as f:
    params = json.load(f)

root_dir = 'results/synchron'

save_str = datetime.datetime(2024, 3, 31)
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
iPert = np.array([0., 0.15, 0.25, 0.35, 0.5])[::-1]

eval_sigma_e = np.array([2.5, 7.5, 13.5]); sige_steps = len(eval_sigma_e)
eval_sigma_i = np.array([4.4, 13.5, 16.5]); sigi_steps = len(eval_sigma_i) 

k_min = -1; k_max = 1; k_steps = 10
eval_k = k_min + np.arange(0, k_steps + 1) / k_steps * (k_max - k_min)

freq_min = 80
freq_max = 140
avalanch_limit = 1.65; conf_int = round(norm.cdf(-avalanch_limit) * 2, 2)

sigVar = np.array([0.01, 0.1, 1, 2])

for nV, _sigvar in enumerate(sigVar):
    for nI, I in enumerate(iPert):
        params['i_e1'] = I / params['gamma'] + params['i_e2']
        print(f"Evaluating for I = {params['i_e1']:.3f}mV")
            
        for k_ind, K in enumerate(eval_k):
            print(f'Running for K = {K:.3f} ----> ', end='')
            params['K'] = K

            AvalanchSync = np.zeros((sige_steps, sigi_steps))
            AvalanchEventDist = np.zeros((sige_steps, sigi_steps))
            KuramotoOrder = np.zeros((sige_steps, sigi_steps))
            nlcor_ = np.zeros((sige_steps, sigi_steps))
            distCorr = np.zeros((sige_steps, sigi_steps))

            GlobalSync = np.zeros((sige_steps, sigi_steps))
            SVDSync = np.zeros((sige_steps, sigi_steps))
            MI_phase = np.zeros((sige_steps, sigi_steps))
            MI_amp = np.zeros((sige_steps, sigi_steps))
            PLVSync = np.zeros((sige_steps, sigi_steps))

            file_name = f"K_{params['K']:.2f}_I_{params['i_e1']:.3f}mV_sigVar_{_sigvar:.2f}.pkl"

            if os.path.isfile(os.path.join(save_dir, file_name)):
                print(f"file found -- skipping processing")
                continue
            
            for se, sige in enumerate(eval_sigma_e):
                tic = time.time()
                for si, sigi in enumerate(eval_sigma_i):
                    params['sig_e1'] = [sige] * n_trials
                    params['sig_i1'] = [sigi] * n_trials

                    # Creating sigma array with covariance
                    params['sig_e2'] = []
                    params['sig_i2'] = []

                    for q in range(n_trials):
                        np.random.seed(q * 10 + 5)
                        sige2 = abs(np.random.normal(sige, _sigvar, 1))
                        sigi2 = abs(np.random.normal(sigi, _sigvar, 1))

                        params['sig_e2'].append(sige2[0])
                        params['sig_i2'].append(sigi2[0])

                    u = np.zeros((n_trials, 4, T))
                    u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

                    # Avalanch synch calculation
                    ASynch = AvalanchSynch(time_series=u, thresh1=avalanch_limit, thresh2=avalanch_limit, tstart=t_start, tend=t_end)
                    dynamic_synch, dynamic_dist = ASynch.run()
                    AvalanchSync[se, si] = np.mean(dynamic_synch)
                    AvalanchEventDist[se, si] = np.mean(dynamic_dist)

                    # Kuramoto Order parameter
                    KuramotoOrder[se, si] = np.mean(kuramoto_order_calc(time_series=u, tstart=t_start, tend=t_end, window_size=window_size, step_window=overlap_size))

                    # Spectral synch calculation
                    SSync = SpectrumSynch(time_series=u, tstart=t_start, tend=t_end, fmin=freq_min, fmax=freq_max, fs=1000, window_size=window_size, step_window=overlap_size)
                    glob_trial, svd_trial, mi_phase_trial, mi_amp_trial, plv_trial = SSync.calc_synchronization()

                    GlobalSync[se, si] = np.mean(glob_trial)
                    SVDSync[se, si] = np.mean(svd_trial)
                    MI_phase[se, si] = np.mean(mi_phase_trial)
                    MI_amp[se, si] = np.mean(mi_amp_trial)
                    PLVSync[se, si] = np.mean(plv_trial)

                    CORR = DynamicCorr(time_series=u, tstart=t_start, tend=t_end, window_size=window_size, step_window=overlap_size)
                    distance_corr, nonlin_corr = CORR.run()

                    distCorr[se, si] = np.mean(distance_corr)
                    nlcor_[se, si] = np.mean(nonlin_corr)

                toc = time.time()
            
                print(f"sigma_e = {sige:.3f} :: {toc - tic}s")

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