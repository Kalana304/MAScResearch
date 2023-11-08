import os
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

from utils import calc_fixed_points, calc_jacobian

plt.rcParams.update({'font.size': 16})

def spectral_rad_analysis(var_min, var_max, d_var, var_name='K', plot_opts=None, if_save=False, save_dir=None,
                          image_name=None, **opts):
    var_arr = np.arange(var_min, var_max, d_var)

    rad_colors = plt.get_cmap('Blues')
    rad_colors = iter(rad_colors(np.linspace(0.2, 1, len(var_arr))))

    eig_val_colors = plt.get_cmap('Reds')
    eig_val_colors = iter(eig_val_colors(np.linspace(0.2, 1, len(var_arr))))

    x = np.arange(-50, 50, 0.01)

    fig, ax = plt.subplots(figsize=plot_opts['fig_size'])
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)

    for iter_no, var in enumerate(var_arr):
        opts[var_name] = var
        fp_dict = calc_fixed_points(
            minval = -1 / opts.get('gamma'),
            maxval = 1 / opts.get('gamma'),
            grid_res = 20,
            n_init = 20,
            cal_grad=False,
            **opts
        )

        eig_dict = calc_jacobian(fp_dict['fps'].T, **opts)
        color_rads = next(rad_colors)
        color_eig = next(eig_val_colors)

        if len(eig_dict['eigvals']) != 0:
            for eig_vals in (eig_dict['eigvals']):
                eig_real = eig_vals.real
                eig_img = eig_vals.imag

                spect_center = np.mean(eig_vals)
                spect_radius = np.abs(eig_vals - spect_center).max()

                ax.plot(eig_real, eig_img, '*', markersize=10, color=color_eig,
                        label=plot_opts['label_tag'] + f' = {var:.3f}')
                ax.plot(x, np.sqrt(spect_radius ** 2 - (x - spect_center.real) ** 2), color=color_rads)
                ax.plot(x, -np.sqrt(spect_radius ** 2 - (x - spect_center.real) ** 2), color=color_rads)

    plt.axvline(x=0, color='k', linestyle='--')
    plt.axis('equal')
    plt.xlabel(r'$\mathfrak{Re}$');
    plt.ylabel(r'$\mathfrak{Im}$')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), fontsize=11, ncols=5, fancybox=True, shadow=True)
    plt.title('Spectral radius analysis with change of ' + plot_opts['label_tag'] + '\n' + plot_opts['para_str'] + '\n')

    if if_save:
        plt.savefig(os.path.join(save_dir, image_name + '.png'), dpi=600, bbox_inches='tight')

    plt.show()
    return