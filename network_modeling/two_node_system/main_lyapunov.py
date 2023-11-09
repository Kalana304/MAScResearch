import os
import glob
import json
import time
import random
import pickle
import datetime
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from scipy import signal, linalg
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

from utils import *
from plot_utils import *
from measure_utils import *

COLOR_MAP = {
                'mediumblue': "Unstable Node",
                'darkgreen': "Stable Node",
                'red': "Unstable Saddle Point",
                'saddlebrown': "Stable Spiral",
                'purple': "Unstable Spiral",
                'black': "Circle"
            }
# grid parameters
minval = -1 / 0.016
maxval = 1 / 0.016
grid_res = 20
n_trials = 5

# Numerical ODE integration to plot trajectories
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)


def euler_intergrate(u, **opts):
    # common parameters
    beta = opts.get('beta')
    gamma = opts.get('gamma')
    theta = opts.get('theta')

    sig_e1 = opts.get('sig_e1')
    sig_e2 = opts.get('sig_e2')
    sig_i1 = opts.get('sig_i1')
    sig_i2 = opts.get('sig_i2')

    dt = opts.get('dt')
    tau_e = opts.get('tau_e')
    tau_i = opts.get('tau_i')

    wee = opts.get('wee')
    wei = opts.get('wei')
    wie = opts.get('wie')
    wii = opts.get('wii')

    i_e1 = opts.get('i_e1')
    i_e2 = opts.get('i_e2')
    i_i1 = opts.get('i_i1')
    i_i2 = opts.get('i_i2')

    T = opts.get('T')
    # print('running euler intergration')

    for k, y0 in enumerate(zip(x0_, x1_, x2_, x3_)):
        # initialize with the points
        u[k, :, 0] = y0
        for t in range(T - 1):
            u[k, 0, t + 1] = u[k, 0, t] + dt / tau_e * \
                             (-u[k, 0, t] + wee * F(u[k, 0, t], beta, theta, gamma, sig_e1) + \
                              wie * F(u[k, 1, t], beta, theta, gamma, sig_i1) + \
                              i_e1 + opts['K'] * u[k, 2, t])

            u[k, 1, t + 1] = u[k, 1, t] + dt / tau_i * \
                             (-u[k, 1, t] + wei * F(u[k, 0, t], beta, theta, gamma, sig_e1) + \
                              wii * F(u[k, 1, t], beta, theta, gamma, sig_i1) + i_i1)

            u[k, 2, t + 1] = u[k, 2, t] + dt / tau_e * \
                             (-u[k, 2, t] + wee * F(u[k, 2, t], beta, theta, gamma, sig_e2) + \
                              wie * F(u[k, 3, t], beta, theta, gamma, sig_i2) + \
                              i_e2 + opts['K'] * u[k, 0, t])

            u[k, 3, t + 1] = u[k, 3, t] + dt / tau_i * \
                             (-u[k, 3, t] + wei * F(u[k, 2, t], beta, theta, gamma, sig_e2) + \
                              wii * F(u[k, 3, t], beta, theta, gamma, sig_i2) + i_i2)
    return u

if __name__ == '__main__':

    load_root = './lyap_experiment_json'
    experiment = ['e1_i1']                    # (e1_i1: changing heterogeneity in node 1)

    # Setting save dir paths
    root_dir = "./sv_coupled_model"

    save_str = datetime.datetime.now()
    sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"


    sigma_arr = np.arange(2.5, 17.5, 1)

    for exp in experiment:
        param_json = glob.glob(os.path.join(load_root, f'{exp}*.json'))

        for iter, param_file_name in enumerate(param_json):
            with open(param_file_name, 'r+') as f:
                print(f'loading parameters from {param_file_name}')
                params = json.load(f)

            save_dir = os.path.join(root_dir, sub_dir, exp)
            print(f"creating dir to save results:: {save_dir}")
            os.makedirs(save_dir, exist_ok=True)

            T = params['T']
            time_ = np.arange(0, T)

            print(f"running {params['version']}")

            # lyapunov exponenet response maps
            lyapunov_n1_mean = np.zeros((len(sigma_arr), len(sigma_arr)))
            lyapunov_n2_mean = np.zeros((len(sigma_arr), len(sigma_arr)))

            lyapunov_n1_max = np.zeros((len(sigma_arr), len(sigma_arr)))
            lyapunov_n2_max = np.zeros((len(sigma_arr), len(sigma_arr)))

            for e1, sigma_e in enumerate(sigma_arr):
                tic = time.time()
                for i1, sigma_i in enumerate(sigma_arr):
                    if exp == 'e1_i1':
                        params['sig_e1'] = sigma_e
                        params['sig_i1'] = sigma_i
                    elif exp == 'e2_i2':
                        params['sig_e2'] = sigma_e
                        params['sig_i2'] = sigma_i

                    u = np.zeros((n_trials, 4, T))
                    u = euler_intergrate(u, **params)

                    lyap_node1 = np.log(np.abs(np.diff(np.mean(u[:, 0, 200:], axis=0))))
                    lyap_node2 = np.log(np.abs(np.diff(np.mean(u[:, 2, 200:], axis=0))))

                    lyapunov_n1_mean[e1, i1] = np.mean(lyap_node1)
                    lyapunov_n2_mean[e1, i1] = np.mean(lyap_node2)
                    lyapunov_n1_max[e1, i1] = lyap_node1.max()
                    lyapunov_n2_max[e1, i1] = lyap_node2.max()

                toc = time.time()
                print(f'finishing sigma value --> Execution time: {toc - tic :.3f}s')


            lyap_matrices = {
                                'n1_mean': lyapunov_n1_mean,
                                'n2_mean': lyapunov_n2_mean,
                                'n1_max': lyapunov_n1_max,
                                'n2_max': lyapunov_n2_max
                            }

            if exp == 'e1_i1':
                para_str = r"($K_{12} = $" + f"{params['K']:.2f}, " + r"$I_{in}^1 = $" + f"{params['i_e1']:.3f}mV, " + \
                           r"$\sigma_e^2$ = " + f"{params['sig_e2']:.2f}, " + r"$\sigma_i^2$ = " + f"{params['sig_i2']:.2f})"
                plot_opts_dict = {
                                    'fig_size': (20, 16),
                                    'ylabel': r'$\sigma_e^1$',
                                    'xlabel': r'$\sigma_i^1$',
                                    'para_str': para_str
                                }
            elif exp == 'e2_i2':
                para_str = r"($K_{12} = $" + f"{params['K']:.2f}, " + r"$I_{in}^1 = $" + f"{params['i_e1']:.3f}mV, " + \
                           r"$\sigma_e^1$ = " + f"{params['sig_e1']:.2f}, " + r"$\sigma_i^1$ = " + f"{params['sig_i1']:.2f})"
                plot_opts_dict = {
                    'fig_size': (20, 16),
                    'ylabel': r'$\sigma_e^2$',
                    'xlabel': r'$\sigma_i^2$',
                    'para_str': para_str
                }

            plot_lyapunov_nodes(
                var1_arr=sigma_arr,
                var2_arr=sigma_arr,
                plot_opts=plot_opts_dict,
                save_dir=save_dir,
                if_save=True,
                image_name=f"lypunov_node_wise_image_{params['version']}",
                **lyap_matrices
            )

            plot_lyapunov_network(
                var1_arr=sigma_arr,
                var2_arr=sigma_arr,
                mask_thresh=0.1,
                plot_opts=plot_opts_dict,
                save_dir=save_dir,
                if_save=True,
                image_name=f"lypunov_network_image_{params['version']}",
                **lyap_matrices
            )

            with open(os.path.join(save_dir, f"lypunov_matrics_{params['version']}.pkl"), 'wb') as f:
                pickle.dump(lyap_matrices, f)
