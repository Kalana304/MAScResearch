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
root_dir = "./sv_coupled_model_m"

save_str = datetime.datetime(2023, 12, 10)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"

save_dir = os.path.join(root_dir, sub_dir)

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
T = params['T']
time_ = np.arange(0, T)

# setting init values
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

t_start = 200
t_end = params['T']

# saving to separate sub-dir
scene_sub_dir = 's_02'
os.makedirs(os.path.join(save_dir, scene_sub_dir), exist_ok=True)

# setting constant values
k_values = np.arange(-1, 1, 0.2)
k_steps = len(k_values)
params['i_e1'] = 0.25 / params['gamma']

# setting variables
sige_min = 2.5; sige_max = 17.5
sige_steps = 100
sigi_min = 2.5; sigi_max = 17.5
sigi_steps = 100

eval_sigma_e = sige_min + np.arange(0, sige_steps) / sige_steps * (sige_max - sige_min)
eval_sigma_i = sigi_min + np.arange(0, sigi_steps) / sigi_steps * (sigi_max - sigi_min)

# lyapunov exponenet response maps sigma_e vs. sigma_i

if_plot_lyap = True
if_save_lyap = True
mask_thresh = 0.05

# spectral radius analysis and its link to stability with lyapunov exponent
spec_rad = np.zeros((k_steps, sige_steps, sigi_steps))
spec_cent = np.zeros((k_steps, sige_steps, sigi_steps))
    
for k_in, k_val in enumerate(k_values):
    tic = time.time()
    params['K'] = k_val
    file_name = f"lyap_K_{params['K']:.2f}_ie_{params['i_e1']:.3f}"

    if os.path.isfile(os.path.join(save_dir, scene_sub_dir, file_name + '.pkl')):
        print(f"file found at {os.path.join(save_dir, scene_sub_dir, file_name + '.pkl')} --> loading file...")
        with open(os.path.join(save_dir, scene_sub_dir, file_name + '.pkl'), 'rb') as f:
            lyap_mats = pickle.load(f)

    else:
        lyap_mats = {
            'n1_mean': np.zeros((sige_steps, sigi_steps)),
            'n2_mean': np.zeros((sige_steps, sigi_steps)),
            'n1_max': np.zeros((sige_steps, sigi_steps)),
            'n2_max': np.zeros((sige_steps, sigi_steps))
        }

        for e1, sigma_e in enumerate(eval_sigma_e):
            params['sig_e1'] = params['sig_e2'] = sigma_e
            tic = time.time()
            for i1, sigma_i in enumerate(eval_sigma_i):
                params['sig_i1'] = params['sig_i2'] = sigma_i

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

                lyap_mats['n1_mean'][e1, i1] = np.mean(lyapunov_1)
                lyap_mats['n2_mean'][e1, i1] = np.mean(lyapunov_2)
                lyap_mats['n1_max'][e1, i1] = lyapunov_1.max()
                lyap_mats['n2_max'][e1, i1] = lyapunov_2.max()
            toc = time.time()
            print(f'finishing sigma value --> Execution time: {toc - tic :.3f}s')

        with open(os.path.join(save_dir, scene_sub_dir, file_name + '.pkl'), 'wb') as f:
            pickle.dump(lyap_mats, f)

    # plotting

    para_str = r"($K = $" + f"{params['K']:.2f}, " + r"$I_{in}^1 = $" + f"{params['i_e1']:.3f}mV)"
    plot_opts_dict = {'fig_size': (10, 10), 'ylabel': r'$\sigma_e$', 'xlabel': r'$\sigma_i$', 'para_str': para_str}

    if if_plot_lyap:
        plot_lyapunov_nodes(
            var1_arr=eval_sigma_e,
            var2_arr=eval_sigma_i,
            plot_opts=plot_opts_dict,
            save_dir=os.path.join(save_dir, scene_sub_dir),
            if_save=if_save_lyap,
            image_name=file_name + '_node',
            **lyap_mats
        )

    thresh_mean, thresh_max = plot_lyapunov_network(
        var1_arr=eval_sigma_e,
        var2_arr=eval_sigma_i,
        mask_thresh=mask_thresh,
        if_plot=if_plot_lyap,
        plot_opts=plot_opts_dict,
        save_dir=os.path.join(save_dir, scene_sub_dir),
        if_save=if_save_lyap,
        image_name=file_name + '_net',
        **lyap_mats
    )

    toc = time.time()

    thresh_node1_mean = lyapunov_threshold(lyap_mats['n1_mean'], mask_tau=mask_thresh)

    # for e1, sig_e in enumerate(eval_sigma_e):
    #     params['sig_e1'] = params['sig_e2'] = sig_e
    #     rad_stable = None
    #     rad_unstable = None
    #     cent_stable = None
    #     cent_unstable = None

    #     tic = time.time()
    #     for i1, sig_i in enumerate(eval_sigma_i):
    #         params['sig_i1'] = params['sig_i2'] = sig_i

    #         iter_count = 0
    #         while iter_count < 100:
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
    #                 spec_rad[k_in, e1, i1] = rad_
    #                 spec_cent[k_in, e1, i1] = center_
    #                 break

    #             iter_count += 1

    #         if fp_dict['n_fps'] == 0:
    #             spec_rad[k_in, e1, i1] = None
    #             spec_cent[k_in, e1, i1] = None
        
    #     toc = time.time()
    print(f"finishing K = {k_val:.2f} Execution time: {toc - tic :.3f}s")

# print("Saving spectral radius files!")
# with open(os.path.join(save_dir, scene_sub_dir, 'spectral_radius_matrix.pkl'), 'wb') as f:
#     pickle.dump(spec_rad, f)

# with open(os.path.join(save_dir, scene_sub_dir, 'spectral_center_matrix.pkl'), 'wb') as f:
#     pickle.dump(spec_cent, f)