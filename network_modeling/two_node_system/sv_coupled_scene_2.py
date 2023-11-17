import os
import json
import time
import random
import pickle
import datetime
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from tqdm import tqdm

from utils import *
from plot_utils import *
from measure_utils import *

plt.rcParams.update({'font.size': 14})

# Setting save dir paths
root_dir = "./sv_coupled_model_m"

save_str = datetime.datetime.now()
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
k_values = [-1, -0.5, -0.2, 0.2, 0.5, 0.8]
params['i_e1'] = 0.25 / params['gamma']

# setting variables
sige_min = 2.5;
sige_max = 17.5;
sige_steps = 100
sigi_min = 2.5;
sigi_max = 17.5;
sigi_steps = 100

eval_sigma_e = sige_min + np.arange(0, sige_steps) / sige_steps * (sige_max - sige_min)
eval_sigma_i = sigi_min + np.arange(0, sigi_steps) / sigi_steps * (sigi_max - sigi_min)

# lyapunov exponenet response maps sigma_e vs. sigma_i

if_plot_lyap = True
if_save_lyap = True
mask_thresh = 0.05

for k_val in k_values:
    params['K'] =k_val
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
            save_dir=save_dir,
            if_save=if_save_lyap,
            image_name=file_name + '_node.png',
            **lyap_mats
        )

    thresh_mean, thresh_max = plot_lyapunov_network(
        var1_arr=eval_sigma_e,
        var2_arr=eval_sigma_i,
        mask_thresh=mask_thresh,
        if_plot=if_plot_lyap,
        plot_opts=plot_opts_dict,
        save_dir=save_dir,
        if_save=if_save_lyap,
        image_name=file_name + '_net.png',
        **lyap_mats
    )

    # spectral radius analysis and its link to stability with lyapunov exponent
    spec_rad = np.zeros((sige_steps, sigi_steps))
    spec_cent = np.zeros((sige_steps, sigi_steps))

    x = np.arange(-50, 50, 0.01)

    for e1, sig_e in enumerate(eval_sigma_e):
        params['sig_e1'] = params['sig_e2'] = sig_e
        rad_stable = None;
        rad_unstable = None
        cent_stable = None;
        cent_unstable = None

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        rad_colors = plt.get_cmap('Reds')
        rad_colors = iter(rad_colors(np.linspace(0.2, 1, sigi_steps)))

        eig_val_colors = plt.get_cmap('Reds')
        eig_val_colors = iter(eig_val_colors(np.linspace(0.2, 1, sigi_steps)))

        file_str = f"K_{params['K']:.3f}_ie1_{params['i_e1']:.3f}_sige_{params['sig_e1']:.3f}"

        tic = time.time()
        for i1, sig_i in enumerate(eval_sigma_i):
            params['sig_i1'] = params['sig_i2'] = sig_i

            fp_dict = calc_fixed_points(
                minval=minval,
                maxval=maxval,
                grid_res=1000,
                n_init=100,
                cal_grad=False,
                **params
            )

            if fp_dict['n_fps'] == 0:
                u = np.zeros((1, 4, T))
                u = euler_intergrate(u, [x0_, x1_, x2_, x3_], **params)

                fe1 = np.max(u[0, 0, t_start:])
                fi1 = np.max(u[0, 1, t_start:])
                fe2 = np.max(u[0, 2, t_start:])
                fi2 = np.max(u[0, 3, t_start:])

                fp_dict['fps'] = np.array([[fe1, fi1, fe2, fi2]]).T
                fp_dict['n_fps'] = 1

            eig_dict = calc_jacobian(fp_dict['fps'].T, **params)
            rad_, center_, real_, img_ = spectrum_analysis(eig_dict['eigvals'])
            spec_rad[e1, i1] = rad_
            spec_cent[e1, i1] = center_

            lyap_val = thresh_mean[e1, i1]

            if lyap_val == 1:
                rad_unstable = rad_
                cent_unstable = center_

            elif lyap_val == -1 or lyap_val == 0:
                rad_stable = rad_
                cent_stable = center_

            color_rads = next(rad_colors)
            color_eig = next(eig_val_colors)

            para_str = r'$K$ = ' + f"{params['K']:.2f}, " + r'$i_{in}^1$ = ' + f"{params['i_e1']:.3f} " + r'$\sigma_e$ = ' + f"{params['sig_e1']:.3f}"
            ax[0].plot(real_, img_, '.', markersize=10, color=color_eig,
                       label=r"$\sigma_i$" + f' = {sig_i:.3f}, r = {rad_:.3f}')
            ax[0].plot(x, np.sqrt(rad_ ** 2 - abs(x - center_) ** 2), color=color_rads)
            ax[0].plot(x, -np.sqrt(rad_ ** 2 - abs(x - center_) ** 2), color=color_rads)

        toc = time.time()
        print(f'finishing sigma value --> Execution time: {toc - tic :.3f}s')

        if (rad_unstable != None) and (rad_stable != None):
            rad_ = (rad_unstable + rad_stable) / 2
            center_ = (cent_unstable + cent_stable) / 2
            ax[0].plot(x, np.sqrt(rad_ ** 2 - abs(x - center_) ** 2), '--', color='g')
            ax[0].plot(x, -np.sqrt(rad_ ** 2 - abs(x - center_) ** 2), '--', color='g')

        elif (rad_unstable == None) and (rad_stable != None):
            rad_ = spec_rad[e1, 0]
            center_ = spec_cent[e1, 0]
            ax[0].plot(x, np.sqrt(rad_ ** 2 - abs(x - center_) ** 2), '--', color='g')
            ax[0].plot(x, -np.sqrt(rad_ ** 2 - abs(x - center_) ** 2), '--', color='g')

        ax[0].axvline(x=0, color='k', linestyle='--')
        ax[0].axis('equal')
        ax[0].grid(True)
        ax[0].set_xlabel(r'$\mathfrak{Re}$');
        ax[0].set_ylabel(r'$\mathfrak{Im}$')

        sns.distplot(spec_rad[e1, :], kde=True, bins=20, hist=False, ax=ax[1])
        _, _, bars = ax[1].hist(spec_rad[e1, :], bins=100, color='red', alpha=0.25, density=True)

        if (rad_unstable != None) and (rad_stable != None):
            ax[1].axvline(x=rad_, color='k', linestyle='--')
            ax[1].set_xlabel('Spectral Radius')
            ax[1].text(x=rad_ + .1, y=.23, s='Unstable', color='red', fontsize=16, animated=True, rotation=90)
            ax[1].text(x=rad_ - .1, y=.23, s='Stable', color='green', fontsize=16, animated=True, rotation=90)
            ax[1].text(x=rad_, y=.33, s=f'{rad_:.3f}', color='k', fontsize=16)

        elif (rad_unstable == None) and (rad_stable != None):
            ax[1].text(x=min(spec_rad[e1, :]), y=.33, s='Whole region is stable', color='green', fontsize=16, animated=True)

        else:
            ax[1].text(x=min(spec_rad[e1, :]), y=.33, s='Whole region is unstable', color='red', fontsize=16, animated=True)

        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.10), ncols=5, fancybox=True, shadow=True)
        plt.suptitle('Spectral radius analysis with change of ' + r"$\sigma_e$ vs.  $\sigma_i$" + '\n' + para_str + '\n')
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, scene_sub_dir, file_str + '.png'), dpi=600, bbox_inches='tight')
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(eval_sigma_e, eval_sigma_i, thresh_mean, cmap='gist_gray', alpha=0.8)
    plt.pcolormesh(eval_sigma_e, eval_sigma_i, spec_rad, cmap='jet', alpha=0.5)
    plt.xlabel(r'$\sigma_i$');
    plt.ylabel(r'$\sigma_e$')
    plt.xticks(eval_sigma_i);
    plt.yticks(eval_sigma_e)
    plt.colorbar()
    plt.title("Spectral radius analysis\n" + r"$K$ = " + f"{params['K']}, " + r"$I_{in}$ = " + f"{params['i_e1']}")
    plt.savefig(os.path.join(save_dir, scene_sub_dir, f"K_{params['K']}_ie_{params['i_e1']}_resmap.png"))
    plt.show()