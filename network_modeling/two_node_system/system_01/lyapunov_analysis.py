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

save_str = datetime.datetime(2024, 3, 4)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"

save_dir = os.path.join(root_dir, type_results, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)

# read parameters from a json file
with open('./params.json', 'r+') as f:
    params = json.load(f)

# setting grid parameters
minval = -5 / params['gamma']
maxval = 5 / params['gamma']
grid_res = 100
n_trials = 5

# time vector
T = params['T'] = 10000
time_ = np.arange(0, T)

# setting init values
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

t_start = 1000
t_end = params['T']

# setting the variables
iPert = [0.0, 0.005, 0.05, 0.5]

sig_min = 2.5; sig_max = 16.5; sig_steps = 20
eval_sigma = sig_min + np.arange(0, sig_steps) / (sig_steps - 1) * (sig_max - sig_min)

k_min = -1; k_max = 1; k_steps = 10
eval_k = k_min + np.arange(0, k_steps) / k_steps * (k_max - k_min)

# lyapunov exponenet response maps K vs. sigma

for nI, I in enumerate(iPert):
    # setting the constant value
    params['i_e1'] = I / params['gamma'] + params['i_e2']
    print(f"Evaluating for I = {I:.4f}")

    file_name = f"lyap_ie_{params['i_e1']:.3f}"

    if_plot_lyap = False
    if_save_lyap = False
    mask_thresh = 0.001

    if os.path.isfile(os.path.join(save_dir, file_name + '.pkl')):
        print(f"file found at {os.path.join(save_dir, file_name + '.pkl')} --> loading file...")

        with open(os.path.join(save_dir, file_name + '.pkl'), 'rb') as f:
            lyap_mats = pickle.load(f)

    else:
        lyap_mats = {
            'n1_mean': np.zeros((k_steps, sig_steps)),
            'n2_mean': np.zeros((k_steps, sig_steps))
        }

        for k1, k in enumerate(eval_k):
            params['K'] = k

            for s1, sigma in enumerate(eval_sigma):
                params['sig_e1'] = params['sig_e2'] = params['sig_i1'] = params['sig_i2'] = sigma

                u = np.zeros((n_trials, 4, T))
                u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

                lyapunov_1 = np.zeros((T - t_start - 1))
                for j in range(n_trials):
                    diff_1 = np.abs(np.diff(u[j, 0, t_start:]))
                    lyapunov_1 += 1 / n_trials * np.log(diff_1)

                lyapunov_2 = np.zeros((T - t_start - 1))
                for j in range(n_trials):
                    diff_2 = np.abs(np.diff(u[j, 2, t_start:]))
                    lyapunov_2 += 1 / n_trials * np.log(diff_2)

                lyap_mats['n1_mean'][k1, s1] = np.mean(lyapunov_1)
                lyap_mats['n2_mean'][k1, s1] = np.mean(lyapunov_2)

        with open(os.path.join(save_dir, file_name + '.pkl'), 'wb') as f:
            pickle.dump(lyap_mats, f)

# plotting

# para_str = r"$I_{in}^1 = $" + f"{params['i_e1']:.3f}mV"
# plot_opts_dict = {'fig_size': (10, 10), 'ylabel': r'$K$', 'xlabel': r'$\sigma$', 'para_str': para_str}

# if if_plot_lyap:
#     plot_lyapunov_nodes(
#         var1_arr=eval_k,
#         var2_arr=eval_sigma,
#         plot_opts=plot_opts_dict,
#         save_dir=os.path.join(save_dir, scene_sub_dir),
#         if_save=if_save_lyap,
#         image_name=file_name + '_node',
#         **lyap_mats
#     )

# thresh_mean, thresh_max = plot_lyapunov_network(
#     var1_arr=eval_k,
#     var2_arr=eval_sigma,
#     mask_thresh=mask_thresh,
#     if_plot=if_plot_lyap,
#     plot_opts=plot_opts_dict,
#     save_dir=os.path.join(save_dir, scene_sub_dir),
#     if_save=if_save_lyap,
#     image_name=file_name + '_net',
#     **lyap_mats
# )

# thresh_node1_mean = lyapunov_threshold(lyap_mats['n1_mean'], mask_tau=mask_thresh)

# # spectral radius analysis and its link to stability with lyapunov exponent

# spec_rad = np.zeros((k_steps, sig_steps))
# spec_cent = np.zeros((k_steps, sig_steps))

# for k1, K in enumerate(eval_k):
#     params['K'] = K
#     rad_stable = None;
#     rad_unstable = None
#     cent_stable = None;
#     cent_unstable = None

#     tic = time.time()
#     for s1, sigma in enumerate(eval_sigma):
#         params['sig_e1'] = params['sig_e2'] = params['sig_i1'] = params['sig_i2'] = sigma

#         iter_count = 0
#         while True:
#             fp_dict = calc_fixed_points(
#                 minval=minval,
#                 maxval=maxval,
#                 grid_res=1000,
#                 n_init=100,
#                 cal_grad=False,
#                 **params
#             )

#             if fp_dict['n_fps'] != 0:
#                 eig_dict = calc_jacobian(fp_dict['fps'].T, **params)
#                 rad_, center_, real_, img_ = spectrum_analysis(eig_dict['eigvals'])
#                 spec_rad[k1, s1] = rad_
#                 spec_cent[k1, s1] = center_
#                 break
#             iter_count += 1
        
#         if fp_dict['n_fps'] == 0:
#             spec_rad[k1, s1] = None
#             spec_cent[k1, s1] = None

#     toc = time.time()
#     print(f'finishing sigma value --> Execution time: {toc - tic :.3f}s')

# print("Saving spectral radius files!")
# with open(os.path.join(save_dir, scene_sub_dir, 'spectral_radius_matrix.pkl'), 'wb') as f:
#     pickle.dump(spec_rad, f)

# with open(os.path.join(save_dir, scene_sub_dir, 'spectral_center_matrix.pkl'), 'wb') as f:
#     pickle.dump(spec_cent, f)