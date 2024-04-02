import os
import json
import time
import random
import pickle
import datetime
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import scipy as sp
from scipy import signal 
from skimage.feature import canny
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator


plt.rcParams.update({'font.size': 14})
THRESH_COLORS = ["aliceblue", "cornflowerblue", "darkblue"]


def plot_ts(u, t_arr, t1, t2, fig_size=(10, 6), if_zoom=False, zoom_t=[], if_save=False, save_dir=None, image_name=None,
            **opts):
    u_plot_arr = u[:, :, t1: t2]
    t_prime = t_arr[: -t1 + t2 - 1]

    title_str = f'Population dynamics with ' + r'$K_{12}$ = ' + f"{opts.get('K'):.2f} and " + r'$I_{in}^1$ = ' + f"{opts.get('i_e1'):.3f} mV"
    sub_title_str = r'($\sigma_e^1$ = ' + \
                    f"{opts.get('sig_e1'):.3f}, " + \
                    r'$\sigma_i^1$ = ' + \
                    f"{opts.get('sig_i1'):.3f}, " + \
                    r'$\sigma_e^2$ = ' + \
                    f"{opts.get('sig_e2'):.3f}, " + \
                    r'$\sigma_i^2$ = ' + \
                    f"{opts.get('sig_i2'):.3f})\n"

    fig, axis = plt.subplots(3, 1, figsize=fig_size, sharex='col')
    fig.subplots_adjust(hspace=.35)

    axis[0].plot(t_arr[: -t1 + t2], np.mean(u_plot_arr[:, 0, :], axis=0), 'red', label='node 1')
    axis[0].plot(t_arr[: -t1 + t2], np.mean(u_plot_arr[:, 2, :], axis=0), 'black', label='node 2')
    axis[0].set_xlim([t_arr[0], t_arr[-t1 + t2 - 1]])
    axis[0].yaxis.set_label_position("right")
    axis[0].set_ylabel(r'$U_e^{1,2}~(a.u.)$');
    axis[0].set_title(f'Excitatory population time series')

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    axis[0].tick_params(axis='y', which='both', labelleft=False, labelright=True)

    axis[1].plot(t_arr[: -t1 + t2], np.mean(u_plot_arr[:, 1, :], axis=0), 'red', label='node 1')
    axis[1].plot(t_arr[: -t1 + t2], np.mean(u_plot_arr[:, 3, :], axis=0), 'black', label='node 1')
    axis[1].set_xlim([t_arr[0], t_arr[-t1 + t2 - 1]])
    axis[1].yaxis.set_label_position("right")
    axis[1].set_ylabel(r'$U_i^{1,2}~(a.u.)$');
    axis[1].set_title(f'Inhibitory population time series')

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    axis[1].tick_params(axis='y', which='both', labelleft=False, labelright=True)

    lyapunov_1 = np.zeros((len(t_prime)))

    for k in range(u_plot_arr.shape[0]):
        diff_1 = np.abs(np.diff(u_plot_arr[k, 0, :]))
        lyapunov_1 += 1 / u_plot_arr.shape[0] * np.log(diff_1)

    lyapunov_2 = np.zeros((len(t_prime)))

    for k in range(u_plot_arr.shape[0]):
        diff_2 = np.abs(np.diff(u_plot_arr[k, 2, :]))
        lyapunov_2 += 1 / u_plot_arr.shape[0] * np.log(diff_2)

    axis[2].plot(t_prime, lyapunov_1, 'red', label='node 1')
    axis[2].plot(t_prime, lyapunov_2, 'black', label='node 2')
    axis[2].plot(t_prime, (lyapunov_1 + lyapunov_2) / 2, 'green', label='network')
    axis[2].set_xlim([t_prime[0], t_prime[- 1]])
    axis[2].yaxis.set_label_position("right")
    axis[2].set_ylabel(r'$\lambda_e^{1,2}$');
    axis[2].set_xlabel(r'$t~(a.u.)$')
    axis[2].set_title(
        f'Lyapunov series for excitatory dynamics (' + r'$<\lambda_e^1>$ = ' + f'{np.mean(lyapunov_1):.3f}, ' + \
        r'$<\lambda_e^2>$ = ' + f'{np.mean(lyapunov_2):.3f}, ' + r'$<\lambda_e>$ = ' + f'{(np.mean(lyapunov_1 + lyapunov_2)) / 2:.3f})'
    )

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    axis[2].tick_params(axis='y', which='both', labelleft=False, labelright=True)

    fig.suptitle(title_str + '\n' + sub_title_str, fontsize=24)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.65), ncols=3, fancybox=True, shadow=True)

    if if_zoom:
        (x0, y0), (x1, y1) = axis[0].get_position().get_points()
        sub_axes_1 = plt.axes([0, y0 + .15, .1, .05])
        sub_axes_1.plot(t_arr[zoom_t[0]: zoom_t[1]], np.mean(u_plot_arr[:, 0, zoom_t[0]:zoom_t[1]], axis=0), c='r')
        sub_axes_1.set_ylabel(r'$U_e^1$')
        plt.setp(sub_axes_1, yticks=[])

        sub_axes_2 = plt.axes([0, y0 - 0.02, .1, .05])
        sub_axes_2.plot(t_arr[zoom_t[2]: zoom_t[3]], np.mean(u_plot_arr[:, 2, zoom_t[2]:zoom_t[3]], axis=0), c='k')
        sub_axes_2.set_ylabel(r'$U_e^2$')
        plt.setp(sub_axes_2, yticks=[], xticks=[])

    plt.tight_layout()

    if if_save:
        fig.savefig(os.path.join(save_dir, image_name + '.png'), dpi=600)
    plt.show()
    return (np.mean(lyapunov_1) + np.mean(lyapunov_2)) / 2


def plot_trajectory(ts, init_points, fig_size=(10, 10), **results):
    dim_labels = [r'$U_e^1$', r'$U_i^1$', r'$U_e^2$', r'$U_i^2$']
    plt_grads = False

    if 'grads' in results.keys():
        plt_grads = True
        grads = results.get('grads')
        locs = results.get('locs')

    fp = results.get('fps')
    fig, axes = plt.subplots(2, 2, figsize=fig_size)

    if plt_grads:
        axes[0, 0].quiver(locs[1][0, 0, :, :], locs[0][0, 0, :, :], grads[1][0, 0, :, :], grads[0][0, 0, :, :]) #e1 v. i1
        axes[0, 1].quiver(locs[3][:, :, 0, 0], locs[2][:, :, 0, 0], grads[3][:, :, 0, 0], grads[2][:, :, 0, 0]) #e2 v. i2
        axes[1, 0].quiver(locs[2][:, :, 0, 0], locs[0][0, 0, :, :], grads[2][:, :, 0, 0], grads[0][0, 0, :, :]) #e2 v. e2
        axes[1, 1].quiver(locs[3][:, :, 0, 0] - locs[1][0, 0, :, :],
                          locs[2][:, :, 0, 0] - locs[0][0, 0, :, :],
                          grads[3][:, :, 0, 0] - grads[1][0, 0, :, :],
                          grads[2][:, :, 0, 0] - grads[0][0, 0, :, :]) #e2-e1 v. i2-i1

    # Plotting the vector field in the state space (E, I)
    axes[0, 0].set_ylabel(dim_labels[0]);
    axes[0, 0].set_xlabel(dim_labels[1])
    axes[0, 0].grid(True)

    axes[0, 1].set_ylabel(dim_labels[2]);
    axes[0, 1].set_xlabel(dim_labels[3])
    axes[0, 1].grid(True)

    axes[1, 0].set_ylabel(dim_labels[0]);
    axes[1, 0].set_xlabel(dim_labels[2])
    axes[1, 0].grid(True)

    axes[1, 1].set_ylabel(f"{dim_labels[2]} - {dim_labels[0]}");
    axes[1, 1].set_xlabel(f"{dim_labels[3]} - {dim_labels[1]}")
    axes[1, 1].grid(True)

    ninits, sys_dim, _ = ts.shape

    x0_, x1_, x2_, x3_ = init_points

    for k, y0 in enumerate(zip(x0_, x1_, x2_, x3_)):
        kts = ts[k, :, :]
        # xdim0, xdim1 = xSolve[plot_dim]

        # Plot the solution in the state space
        axes[0, 0].plot(kts[1, :], kts[0, :], '-')
        axes[0, 0].scatter(y0[1], y0[0], marker='.', c='r', s=100, label=f"{dim_labels[0]} = {y0[0]:.3f}, {dim_labels[1]} = {y0[1]:.3f}")

        axes[0, 1].plot(kts[3, :], kts[2, :], '-')
        axes[0, 1].scatter(y0[3], y0[2], marker='.', c='r', s=100,
                           label=f"{dim_labels[2]} = {y0[2]:.3f}, {dim_labels[3]} = {y0[3]:.3f}")

        axes[1, 0].plot(kts[2, :], kts[0, :], '-')
        axes[1, 0].scatter(y0[2], y0[0], marker='.', c='r', s=100,
                           label=f"{dim_labels[0]} = {y0[0]:.3f}, {dim_labels[2]} = {y0[2]:.3f}")

        axes[1, 1].plot(kts[3, :] - kts[1, :], kts[2, :] - kts[0, :], '-')
        axes[1, 1].scatter(y0[3] - y0[1], y0[2] - y0[0], marker='.', c='r', s=100,
                           label=f"{dim_labels[2]} - {dim_labels[0]} = {y0[2] - y0[0]:.3f}, {dim_labels[3]} - {dim_labels[1]} = {y0[3] - y0[1]:.3f}")

    # Plot the fixed points identified
    if len(fp) > 0:
        axes[0, 0].scatter(fp[1], fp[0], marker='o', c='k', s=100, label=f"Fixed points")
        axes[0, 1].scatter(fp[3], fp[2], marker='o', c='k', s=100, label=f"Fixed points")
        axes[1, 0].scatter(fp[2], fp[0], marker='o', c='k', s=100, label=f"Fixed points")
        axes[1, 1].scatter(fp[3] - fp[1], fp[2] - fp[0], marker='o', c='k', s=100, label=f"Fixed points")


    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)

    axes[0, 1].tick_params(axis='y', which='both', labelleft=False, labelright=True)
    axes[1, 1].tick_params(axis='y', which='both', labelleft=False, labelright=True)
    fig.suptitle('Phase plane projections')
    fig.tight_layout()
    plt.show()
    return


# plotting each node's lyapunov exponents (time-averaged and maximum)
def plot_lyapunov_nodes(var1_arr, var2_arr, plot_opts=None, save_dir=None, if_save=False, image_name=False,
                        **lyap_mat):
    plt.figure(figsize=plot_opts['fig_size'])
    plt.subplot(2, 2, 1)

    plt.pcolormesh(var2_arr, var1_arr, lyap_mat.get('n1_mean'), cmap='jet')
    plt.ylabel(plot_opts['ylabel']); plt.xlabel(plot_opts['xlabel'])
    plt.xticks(var2_arr, rotation=90); plt.yticks(var1_arr)
    plt.title(r'$\left<\lambda_e^1 \right>_t$')
    plt.colorbar()

    plt.subplot(2, 2, 2)

    plt.pcolormesh(var2_arr, var1_arr, lyap_mat.get('n2_mean'), cmap='jet')
    plt.ylabel(plot_opts['ylabel']); plt.xlabel(plot_opts['xlabel'])
    plt.xticks(var2_arr, rotation=90); plt.yticks(var1_arr)
    plt.title(r'$\left<\lambda_e^2 \right>_t$')
    plt.colorbar()

    plt.subplot(2, 2, 3)

    plt.pcolormesh(var2_arr, var1_arr, lyap_mat.get('n1_max'), cmap='jet')
    plt.ylabel(plot_opts['ylabel']); plt.xlabel(plot_opts['xlabel'])
    plt.xticks(var2_arr, rotation=90); plt.yticks(var1_arr)
    plt.title(r'$\left(\lambda_e^1 \right)_{max}$')
    plt.colorbar()

    plt.subplot(2, 2, 4)

    plt.pcolormesh(var2_arr, var1_arr, lyap_mat.get('n2_max'), cmap='jet')
    plt.ylabel(plot_opts['ylabel']); plt.xlabel(plot_opts['xlabel'])
    plt.xticks(var2_arr, rotation=90); plt.yticks(var1_arr)
    plt.title(r'$\left(\lambda_e^2 \right)_{max}$')
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


def plot_lyapunov_network(var1_arr, var2_arr, mask_thresh=0.1, if_plot=True, plot_opts=None, save_dir=None,
                          if_save=False,
                          image_name=False, **lyap_mat):
    # plotting lyapunov exponents averaged across the network
    lyap_network_mean = (lyap_mat.get('n1_mean') + lyap_mat.get('n2_mean')) / 2
    lyap_network_max = (lyap_mat.get('n1_max') + lyap_mat.get('n2_max')) / 2

    # thresholding for better region understanding
    lyap_mean_thresh = lyapunov_threshold(lyap_network_mean, mask_tau=mask_thresh)
    lyap_max_thresh = lyapunov_threshold(lyap_network_max, mask_tau=mask_thresh)

    if if_plot:
        plt.figure(figsize=plot_opts['fig_size'])
        plt.subplot(2, 2, 1)
        plt.pcolormesh(var2_arr, var1_arr, lyap_network_mean, cmap='jet')
        plt.ylabel(plot_opts['ylabel']);
        plt.xlabel(plot_opts['xlabel'])
        plt.xticks(var2_arr, rotation=90);
        plt.yticks(var1_arr)
        plt.title(r"$\left<\lambda_e\right>_t$")
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.pcolormesh(var2_arr, var1_arr, lyap_network_max, cmap='jet')
        plt.ylabel(plot_opts['ylabel']);
        plt.xlabel(plot_opts['xlabel'])
        plt.xticks(var2_arr, rotation=90);
        plt.yticks(var1_arr)
        plt.title(r"$\left(\lambda_e\right)_{max}$")
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.pcolormesh(var2_arr, var1_arr, lyap_mean_thresh, cmap='jet', vmax=1, vmin=-1)
        plt.ylabel(plot_opts['ylabel']);
        plt.xlabel(plot_opts['xlabel'])
        plt.xticks(var2_arr, rotation=90);
        plt.yticks(var1_arr)
        plt.title(r"$\left<\lambda_e\right>_t$" + f' (thresholded)')
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.pcolormesh(var2_arr, var1_arr, lyap_max_thresh, cmap='jet', vmax=1, vmin=-1)
        plt.ylabel(plot_opts['ylabel']);
        plt.xlabel(plot_opts['xlabel'])
        plt.xticks(var2_arr, rotation=90);
        plt.yticks(var1_arr)
        plt.title(r"$\left(\lambda_e\right)_{max}$" + f' (thresholded)')
        plt.colorbar()

        plt.suptitle("Network stability analysis using Lypunov exponents\n" + plot_opts['para_str'])
        plt.tight_layout()

        if if_save:
            plt.savefig(os.path.join(save_dir, image_name + '.png'), dpi=600, bbox_inches='tight')
        plt.show()

    return lyap_mean_thresh, lyap_max_thresh


def plot_fp_projections(plot_fps_dict, plot_dict=None, save_dir=None, if_save=False, image_name=None, **opts):
    fig, axs = plt.subplots(2, 2, figsize=plot_dict['fig_size'], sharex=True, sharey=True)
    low_xlim = plot_fps_dict['fp_var'][0]
    high_xlim = plot_fps_dict['fp_var'][-1]

    axs[0][0].scatter(plot_fps_dict['fp_var'], plot_fps_dict['fps']['ue1'], marker='.',
                      c=plot_fps_dict['fp_types']['ue1'])
    axs[0][0].set_xlim([low_xlim, high_xlim])
    axs[0][0].set_ylabel(r"$u_e^1$");
    # axs[0][0].set_xlabel(plot_dict['xlabel'])
    axs[0][0].set_title("Node 01 (excitatory)");
    axs[0][0].grid(True)

    axs[0][1].scatter(plot_fps_dict['fp_var'], plot_fps_dict['fps']['ui1'], marker='.',
                      c=plot_fps_dict['fp_types']['ui1'])
    axs[0][1].set_xlim([low_xlim, high_xlim])
    axs[0][1].set_ylabel(r"$u_i^1$");
    # axs[0][1].set_xlabel(plot_dict['xlabel'])
    axs[0][1].set_title("Node 01 (inhibitory)");
    axs[0][1].grid(True)

    axs[1][0].scatter(plot_fps_dict['fp_var'], plot_fps_dict['fps']['ue2'], marker='.',
                      c=plot_fps_dict['fp_types']['ue2'])
    axs[1][0].set_xlim([low_xlim, high_xlim])
    axs[1][0].set_ylabel(r"$u_e^2$");
    # axs[1][0].set_xlabel(plot_dict['xlabel'])
    axs[1][0].set_title("Node 02 (excitatory)");
    axs[1][0].grid(True)

    axs[1][1].scatter(plot_fps_dict['fp_var'], plot_fps_dict['fps']['ui2'], marker='.',
                      c=plot_fps_dict['fp_types']['ui2'])
    axs[1][1].set_xlim([low_xlim, high_xlim])
    axs[1][1].set_ylabel(r"$u_i^2$");
    # axs[1][1].set_xlabel(plot_dict['xlabel'])
    axs[1][1].set_title("Node 02 (inhibitory)");
    axs[1][1].grid(True)

    # fig.suptitle(f"Fixed point evolution with {plot_dict['var_name']}\n" + f"{plot_dict['d_var_str']}")
    fig.supxlabel(plot_dict['xlabel'])

    fig.tight_layout()

    if if_save:
        fig.savefig(os.path.join(save_dir, os.path.basename(image_name) + '.png'), dpi = 600, bbox_inches='tight')
    plt.show()

    return


def wavelet_analysis(signals, time_arr, t1, t2, center_freq=12., fs=1000, fig_size=(24, 8), save_dir = None, if_plot=False, if_save=False, img_name=None, **opts):
    freq_arr = np.linspace(1, fs / 2, 2000)
    widths = center_freq * fs / (2 * freq_arr * np.pi)

    wtmatr = signal.cwt(signals[t1:t2] - np.mean(signals[t1:t2]), signal.morlet2, widths, w=center_freq)
    wtmatr = np.flipud(wtmatr)

    if if_plot:
        plt.figure(figsize=fig_size)
        plt.subplot(3, 2, 1)
        plt.plot(time_arr[t1:t2], signals[t1:t2])
        plt.xlabel('Time (au)')
        plt.ylabel(r'$U_e$')
        plt.xlim(time_arr[t1], time_arr[t2])
        plt.title('Time Series')

        plt.subplot(1, 2, 2)
        plt.imshow(np.abs(wtmatr) / np.max(np.abs(wtmatr)), extent=[t1, t2, freq_arr[0], freq_arr[-1]],
                   cmap='jet', aspect='auto')
        plt.colorbar()
        plt.ylabel('Freq. (Hz)')
        plt.xlabel('Time (au)')
        plt.title('Wavelet Analysis - Complex Morlet')

        plt.subplot(3, 2, 3)
        plt.specgram(signals[t1:t2], Fs=fs, cmap='jet')
        plt.xlabel('Time (au)')
        plt.ylabel('Freq. (Hz)')
        plt.title('STFT Analysis (PSD)')

        plt.subplot(3, 2, 5)
        plt.specgram(signals[t1:t2], Fs=fs, mode='phase', cmap='jet')
        plt.xlabel('Time (au)')
        plt.ylabel('Freq. (Hz)')
        plt.title('STFT Analysis (Phase)')

        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)

        plt.suptitle(opts.get('suptitle'))
        plt.tight_layout()

        if if_save:
            plt.savefig(os.path.join(save_dir, img_name + '.png'), dpi=600, bbox_inches='tight')

        plt.show()

    return freq_arr, wtmatr / np.max(np.abs(wtmatr))

def lyapunov_upsample(orig_response, eval_arrx, eval_arry, y_min, y_max, x_min, x_max,
                         sigma_x = 25, sigma_y = 25,  resolution = 500):
    """
        This function is to upsample the response maps obtained for Lyapunov stability to a given
        resolution. Further, this smoothens response map to remove the staircase effects during the
        low resolution using a 2-dimensional Gaussian kernel.

        parameters:
            orig_response: (ndarray) the low resolution response map of size (K x K)
            eval_arrx:     (ndarray) the x-variable values from the original resolution
            eval_arry:     (ndarray) the y-variable values from the original resolution
            y_min:         (float) minimum y-variable value
            y_max:         (float) maximum y-variable value
            x_min:         (float) minimum x-variable value
            x_max:         (float) maximum x-variable value
            sigma_x:       (int) kernal width of Gaussian kernel
            sigma_y:       (int) kernel height of Gaussian kernel
            resolution:    (int) target resolution of the resulting response map
        
        return:
            high_res_y:            (ndarray) the y-variable values from the high resolution
            high_res_x:            (ndarray) the x-variable values from the high resolution
            smoothed_node_01_high: (ndarray) the high resolution response map of size (resolution x resolution)
    """
    # Upsampling the response map to 500 points
    high_res_y = np.linspace(y_min, y_max, resolution)
    high_res_x = np.linspace(x_min, x_max, resolution)

    fit_points = [eval_arry, eval_arrx]
    interp = RegularGridInterpolator(fit_points, orig_response, bounds_error = False, fill_value = -np.inf)

    eval_y_high, eval_x_high = np.meshgrid(high_res_y, high_res_x, indexing='ij')

    interp_points = np.array([eval_y_high.ravel(), eval_x_high.ravel()]).T

    response_node_01_high = interp(interp_points, method='slinear').reshape(resolution, resolution)
    response_node_01_high = lyapunov_threshold(response_node_01_high, 0.000)

    # Smoothing the response map to have continuous boundary by applying gaussian filter
    sigma = [sigma_y, sigma_x]
    smoothed_node_01_high = sp.ndimage.filters.gaussian_filter(response_node_01_high, sigma, mode='nearest')
    smoothed_node_01_high = lyapunov_threshold(smoothed_node_01_high, 0.000)

    return high_res_y, high_res_x, smoothed_node_01_high

def cv_upsample(orig_response, eval_arrx, eval_arry, y_min, y_max, x_min, x_max,
                         sigma_x = 25, sigma_y = 25,  resolution = 500):
    """
        This function is to upsample the response maps obtained for Coefficient of Variation to a given
        resolution. Further, this smoothens response map to remove the staircase effects during the
        low resolution using a 2-dimensional Gaussian kernel.

        parameters:
            orig_response: (ndarray) the low resolution response map of size (K x K)
            eval_arrx:     (ndarray) the x-variable values from the original resolution
            eval_arry:     (ndarray) the y-variable values from the original resolution
            y_min:         (float) minimum y-variable value
            y_max:         (float) maximum y-variable value
            x_min:         (float) minimum x-variable value
            x_max:         (float) maximum x-variable value
            sigma_x:       (int) kernal width of Gaussian kernel
            sigma_y:       (int) kernel height of Gaussian kernel
            resolution:    (int) target resolution of the resulting response map
        
        return:
            high_res_y:            (ndarray) the y-variable values from the high resolution
            high_res_x:            (ndarray) the x-variable values from the high resolution
            smoothed_node_01_high: (ndarray) the high resolution response map of size (resolution x resolution)
    """
    # Upsampling the response map to 500 points

    high_res_y = np.linspace(y_min, y_max, resolution)
    high_res_x = np.linspace(x_min, x_max, resolution)

    fit_points = [eval_arry, eval_arrx]
    interp = RegularGridInterpolator(fit_points, orig_response, bounds_error = True, fill_value = 0)

    eval_y_high, eval_x_high = np.meshgrid(high_res_y, high_res_x, indexing='ij')

    interp_points = np.array([eval_y_high.ravel(), eval_x_high.ravel()]).T
    response_node_01_high = interp(interp_points, method='slinear').reshape(resolution, resolution)

    # Smoothing the response map to have continuous boundary by applying gaussian filter
    sigma = [sigma_y, sigma_x]
    smoothed_node_01_high = sp.ndimage.filters.gaussian_filter(response_node_01_high, sigma, mode='nearest')

    return high_res_y, high_res_x, smoothed_node_01_high

def boundary_detection(response_map, y_arr, x_arr):
    # edge detection using Canny Edge Detector
    edge_coord = canny(response_map)
    y_coord, x_coord = np.where(edge_coord == 1)

    y_edges = y_arr[y_coord]
    x_edges = x_arr[x_coord]

    # sorting the values
    x_order_ind = np.argsort(x_edges)
    x_edges_sort = x_edges[x_order_ind]
    y_edges_sort = y_edges[x_order_ind]

    # interpolating the points to find the boundary
    function = interp1d(x_edges_sort, y_edges_sort, kind = 'nearest', fill_value='extrapolate')
    
    return function, x_edges_sort, y_edges_sort

def boundary_detection(response_map, y_arr, x_arr):
    """
        This function detects the state transition boundary between stable and unstable
        given a thresholded response map.

        parameters:
            response_map:   (ndarray) the response map of the measured metric
            y_arr:          (ndarray) y-variable array of values
            x_arr:          (ndarray) x-variable array of values
        
        returns:
            function:       (function) an interpolating function learnt from the boundary coordinates
            x_edges_sort:   (ndarray) x-variable values of the boundary
            y_edges_sort:   (ndarray) y-variable values of the boundary

    """
    # edge detection using Canny Edge Detector
    edge_coord = canny(response_map)
    y_coord, x_coord = np.where(edge_coord == 1)

    y_edges = y_arr[y_coord]
    x_edges = x_arr[x_coord]

    # sorting the values
    x_order_ind = np.argsort(x_edges)
    x_edges_sort = x_edges[x_order_ind]
    y_edges_sort = y_edges[x_order_ind]

    if len(x_edges_sort) == 0:
        return None, None, None

    # interpolating the points to find the boundary
    function = interp1d(x_edges_sort, y_edges_sort, kind = 'nearest', fill_value='extrapolate')
    
    return function, x_edges_sort, y_edges_sort