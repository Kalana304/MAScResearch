import os
import json
import time
import random
import pickle
import datetime
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy import signal, linalg
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 16})
THRESH_COLORS = ["aliceblue", "cornflowerblue", "darkblue"]

def plot_ts(u, t_arr, t1, t2, fig_size=(10, 6), if_save=False, save_dir=None, image_name=None, **opts):
    u_plot_arr = u[:, :, t1: t2]
    t_prime = t_arr[t1: t2 - 1]

    title_str = f'Population dynamics with ' + r'$K_{12}$ = ' + f"{opts.get('K'):.2f} and " + r'$I_{in}^1$ = ' + f"{opts.get('i_e1'):.3f} mV"
    sub_title_str = r'($\sigma_e^1$ = ' + \
                    f"{opts.get('sig_e1'):.3f}, " + \
                    r'$\sigma_i^1$ = ' + \
                    f"{opts.get('sig_i1'):.3f}, " + \
                    r'$\sigma_e^2$ = ' + \
                    f"{opts.get('sig_e2'):.3f}, " + \
                    r'$\sigma_i^2$ = ' + \
                    f"{opts.get('sig_i2'):.3f})\n"

    plt.figure(figsize=fig_size)
    plt.subplot(3, 1, 1)
    plt.plot(t_arr[t1: t2], np.mean(u_plot_arr[:, 0, :], axis=0), 'red', label='node 1')
    plt.plot(t_arr[t1: t2], np.mean(u_plot_arr[:, 2, :], axis=0), 'black', label='node 2')
    plt.xlim([t_arr[t1], t_arr[t2 - 1]])
    plt.ylabel(r'$U_e^{1,2}~(a.u.)$');
    plt.xlabel(r'$t~(a.u.)$')
    plt.title(f'Excitatory population time series')

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)

    plt.subplot(3, 1, 2)
    plt.plot(t_arr[t1: t2], np.mean(u_plot_arr[:, 1, :], axis=0), 'red', label='node 1')
    plt.plot(t_arr[t1: t2], np.mean(u_plot_arr[:, 3, :], axis=0), 'black', label='node 1')
    plt.xlim([t_arr[t1], t_arr[t2 - 1]])
    plt.ylabel(r'$U_i^{1,2}~(a.u.)$');
    plt.xlabel(r'$t~(a.u.)$')
    plt.title(f'Inhibitory population time series')

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)

    plt.subplot(3, 1, 3)
    lyapunov_1 = np.mean(u_plot_arr[:, 0, :], axis=0)
    lyapunov_1 = np.log(np.abs(np.diff(lyapunov_1)))

    lyapunov_2 = np.mean(u_plot_arr[:, 2, :], axis=0)
    lyapunov_2 = np.log(np.abs(np.diff(lyapunov_2)))

    plt.plot(t_prime, lyapunov_1, 'red', label='node 1')
    plt.plot(t_prime, lyapunov_2, 'black', label='node 2')
    plt.xlim([t_arr[t1], t_arr[t2 - 1]])
    plt.ylabel(r'$\lambda_e^{1,2}$');
    plt.xlabel(r'$t~(a.u.)$')
    plt.title(
        f'Lyapunov series for excitatory dynamics (' + r'$<\lambda_e^1>$ = ' + f'{np.mean(lyapunov_1):.3f}, ' + r'$<\lambda_e^2>$ = ' + f'{np.mean(lyapunov_2):.3f})')

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)

    plt.suptitle(title_str + '\n' + sub_title_str, fontsize=24)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.85), ncols=2, fancybox=True, shadow=True)
    plt.tight_layout()

    if if_save:
        plt.savefig(os.path.join(save_dir, image_name + '.png'), dpi=600)
    plt.show()
    return


def plot_trajectory(minval, maxval, ode_solves, init_points, plot_dim, fig_size=(5, 5), **results):
    dim_labels = [r'$U_e^1$', r'$U_i^1$', r'$U_e^2$', r'$U_i^2$']
    assert type(plot_dim) == list and len(plot_dim) == 2, "Wrong dims to plot"

    plot_1 = plot_dim[0]
    plot_2 = plot_dim[1]

    dim0 = results.get('locs')[plot_1]
    dim1 = results.get('locs')[plot_2]
    grad0 = results.get('grads')[plot_1]
    grad1 = results.get('grads')[plot_2]
    fp = results.get('fps')

    if (plot_1 == 0 or plot_1 == 1):
        dim0 = dim0[:, :, 0, 0]
        grad0 = grad0[:, :, 0, 0]
    elif (plot_1 == 2 or plot_1 == 3):
        dim0 = dim0[0, 0, :, :]
        grad0 = grad0[0, 0, :, :]

    if (plot_2 == 0 or plot_2 == 1):
        dim1 = dim1[:, :, 0, 0]
        grad1 = grad1[:, :, 0, 0]
    elif (plot_2 == 2 or plot_2 == 3):
        dim1 = dim1[0, 0, :, :]
        grad1 = grad1[0, 0, :, :]

    xlabel = dim_labels[plot_2]
    ylabel = dim_labels[plot_1]

    # Plotting the vector field in the state space (E, I)
    plt.figure(figsize=fig_size)
    plt.quiver(dim1, dim0, grad1, grad0, pivot='mid', alpha=.8)
    plt.xlim([minval, maxval]);
    plt.ylim([minval, maxval])
    plt.xlabel(xlabel);
    plt.ylabel(ylabel)
    plt.grid()

    ninits, sys_dim, _ = ode_solves.shape

    x0_, x1_, x2_, x3_ = init_points

    for k, y0 in enumerate(zip(x0_, x1_, x2_, x3_)):
        xSolve = ode_solves[k, :, :]
        xdim0, xdim1 = xSolve[plot_dim]

        # Plot the solution in the state space
        plt.plot(xdim1, xdim0, '-', )

        # Plot the starting point
        plt.scatter(y0[plot_2], y0[plot_1], marker='*', c='r', s=300,
                    label=f"{ylabel} = {y0[plot_1]:.3f} {xlabel} = {y0[plot_2]:.3f}")

    # Plot the fixed points identified
    plt.scatter(fp[plot_2], fp[plot_1], marker='o', c='k', s=100, label="Stationary points")
    plt.legend(loc='center right', bbox_to_anchor=(-0.1, 0.5), ncols=1, fancybox=True, shadow=True, fontsize=12)

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)

    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)

    plt.show()
    return


# plotting each node's lyapunov exponents (time-averaged and maximum)
def plot_lyapunov_nodes(var1_arr, var2_arr, plot_opts=None, save_dir=None, if_save=False, image_name=False,
                        **lyap_mat):
    plt.figure(figsize=plot_opts['fig_size'])
    plt.subplot(2, 2, 1)

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)

    plt.pcolormesh(var2_arr, var1_arr, lyap_mat.get('n1_mean'), cmap='Blues')
    plt.ylabel(plot_opts['ylabel']);
    plt.xlabel(plot_opts['xlabel'])
    plt.title('Time-averaged lyapunov exponents (node 01)')
    plt.colorbar()

    plt.subplot(2, 2, 2)

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.pcolormesh(var2_arr, var1_arr, lyap_mat.get('n2_mean'), cmap='Blues')
    plt.ylabel(plot_opts['ylabel']);
    plt.xlabel(plot_opts['xlabel'])
    plt.title('Time-averaged lyapunov exponents (node 02)')
    plt.colorbar()

    plt.subplot(2, 2, 3)

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.pcolormesh(var2_arr, var1_arr, lyap_mat.get('n1_max'), cmap='Blues')
    plt.ylabel(plot_opts['ylabel']);
    plt.xlabel(plot_opts['xlabel'])
    plt.title('Max. lyapunov exponents (node 01)')
    plt.colorbar()

    plt.subplot(2, 2, 4)

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.pcolormesh(var2_arr, var1_arr, lyap_mat.get('n2_max'), cmap='Blues')
    plt.ylabel(plot_opts['ylabel']);
    plt.xlabel(plot_opts['xlabel'])
    plt.title('Max. lyapunov exponents (node 02)')
    plt.colorbar()

    plt.suptitle("Node-wise stability analysis using lypunov exponents\n" + plot_opts['para_str'])
    plt.tight_layout()

    if if_save:
        plt.savefig(os.path.join(save_dir, image_name + '.png'), dpi=600, bbox_inches='tight')
    plt.show()
    return

def lyapunov_threshold(raw_data, mask_tau=0.05):
    results = deepcopy(raw_data)
    zero_mask = (results >= -mask_tau) & (results <= mask_tau)
    results[results > mask_tau] = 1
    results[results < -mask_tau] = -1
    results[zero_mask] = 0

    return results

def plot_lyapunov_network(var1_arr, var2_arr, mask_thresh=0.1, plot_opts=None, save_dir=None, if_save=False,
                          image_name=False, **lyap_mat):
    # plotting lyapunov exponents averaged across the network
    lyapunov_network_mean = (lyap_mat.get('n1_mean') + lyap_mat.get('n2_mean')) / 2
    lyapunov_network_max = (lyap_mat.get('n1_max') + lyap_mat.get('n2_max')) / 2

    # thresholding for better region understanding
    lyapunov_mean_thresh = lyapunov_threshold(lyapunov_network_mean, mask_tau=mask_thresh)
    lyapunov_max_thresh = lyapunov_threshold(lyapunov_network_max, mask_tau=mask_thresh)

    plt.figure(figsize=plot_opts['fig_size'])
    plt.subplot(2, 2, 1)
    plt.pcolormesh(var2_arr, var1_arr, lyapunov_network_mean, cmap='Blues')
    plt.ylabel(plot_opts['ylabel']);
    plt.xlabel(plot_opts['xlabel'])
    plt.title('Mean lyapunov exponents (avg. across network)')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.pcolormesh(var2_arr, var1_arr, lyapunov_network_max, cmap='Blues')
    plt.ylabel(plot_opts['ylabel']);
    plt.xlabel(plot_opts['xlabel'])
    plt.title('Max lyapunov exponents (avg. across network)')
    plt.colorbar()

    cmap = ListedColormap(THRESH_COLORS)

    plt.subplot(2, 2, 3)
    plt.pcolormesh(var2_arr, var1_arr, lyapunov_mean_thresh, cmap=cmap)
    plt.ylabel(plot_opts['ylabel']);
    plt.xlabel(plot_opts['xlabel'])
    plt.title('Mean lyapunov exponents (avg. across network) - thresholded')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.pcolormesh(var2_arr, var1_arr, lyapunov_max_thresh, cmap=cmap)
    plt.ylabel(plot_opts['ylabel']);
    plt.xlabel(plot_opts['xlabel'])
    plt.title('Max lyapunov exponents (avg. across network) - thresholded')
    plt.colorbar()

    plt.suptitle("Network stability analysis using lypunov exponents\n" + plot_opts['para_str'])
    plt.tight_layout()

    if if_save:
        plt.savefig(os.path.join(save_dir, image_name + '.png'), dpi=600, bbox_inches='tight')
    plt.show()

    return