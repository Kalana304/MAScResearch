import os
import glob
import json
import pickle
import datetime
import itertools
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import networkx as nx

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import scipy as sp
from scipy import signal 
from skimage.feature import canny
from scipy.interpolate import interp1d
from sklearn.decomposition import KernelPCA
from scipy.interpolate import RegularGridInterpolator

plt.rcParams.update({'font.size': 14})

CORRTAGS = {
            'DCOR': r'$D_{corr}$',
            'CORCOEF': r'$C_{corr}$',
            'PLV_LOW': r'$PLV_{corr}^L$',
            'PLV_HIGH': r'$PLV_{corr}^H$',
            'PLI': r'$PLI_{corr}$',
            'COHERENCE_LOW': r'$Coh_{corr}^L$',
            'COHERENCE_HIGH': r'$Coh_{corr}^H$',
}

def FCProgressPlot(corrMat, ROI_num, corr_type, SaveDir, ifSave=False, figsize=(35, 5), **opts):
    fig, axs = plt.subplots(1, 8, figsize=figsize, layout='constrained')

    total_corr = 0
    nNodes = len(ROI_num)
    corrMat[np.isnan(corrMat)] = 0

    if corr_type == 'CORCOEF':
        corrMat[corrMat < 0] = 0

    for i in range(corrMat.shape[0]):
        corrMat[i] = corrMat[i] - np.diag(np.diag(corrMat[i]))          # remove self connections

        G = nx.from_numpy_array(corrMat[i], parallel_edges=False)
        pos = nx.circular_layout(G) 
        nx.draw_networkx_nodes(G, pos, ax=axs[i], node_color='lightblue', edgecolors='darkblue', node_size=200)

        # Adding labels to the nodes
        labels = {}
        for n in range(nNodes):
            labels[n] = f"{ROI_num[n]:02d}"
        nx.draw_networkx_labels(G, pos,labels, ax=axs[i], font_size=8)

        all_weights = []
        #4 a. Iterate through the graph nodes to gather all the weights
        for (node1, node2, data) in G.edges(data = True):
            all_weights.append(data['weight']) 

        #4 b. Get unique weights
        unique_weights = list(set(all_weights))

        if (len(unique_weights) == 0) or ((len(unique_weights) == 1) and unique_weights[0] == 0):
            #Plot the graph
            axs[i].axis('off')
            axs[i].set_title(f"{np.sum(corrMat[i][np.triu_indices(nNodes, k=1)]) / (nNodes * (nNodes - 1) / 2):2f}")

            continue

        maxWeight = np.max(unique_weights)

        #4 c. Plot the edges - one by one!
        for weight in unique_weights:
            #4 d. Form a filtered list with just the weight you want to draw
            weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
            nx.draw_networkx_edges(G, pos, ax=axs[i], edge_color='darkblue', alpha = weight / maxWeight, edgelist = weighted_edges, width = 2.5)

        #Plot the graph
        axs[i].axis('off')
        axs[i].set_title(f"{np.sum(corrMat[i][np.triu_indices(nNodes, k=1)]) / (nNodes * (nNodes - 1) / 2):2f}", fontsize=20)
        total_corr += (1 / corrMat.shape[0]) * np.sum(corrMat[i][np.triu_indices(nNodes, k=1)]) / (nNodes * (nNodes - 1) / 2)

    fig.text(1.02, 0.8, r'$\sigma_e = $' + f"{opts['sigma'][0]:.3f}", fontsize=25)
    fig.text(1.02, 0.65, r'$\sigma_i = $' + f"{opts['sigma'][1]:.3f}", fontsize=25)
    fig.text(1.02, 0.5, r'$K_{glob} = $' + f"{opts['Kglob']:.2f}", fontsize=25)
    fig.text(1.02, 0.35, r'$I^m_{e} = $' + f"{opts['i_mod']:.3f}mV", fontsize=25)
    fig.text(1.02, 0.2, CORRTAGS[corr_type] + f" = {total_corr:.3f}", fontsize=25)

    if ifSave and SaveDir:
        fileName = f"{corr_type}_sige_{opts['sigma'][0]:.3f}_sigi_{opts['sigma'][1]:.3f}_Imod_{opts['i_mod']:.3f}mV_KGlob_{opts['Kglob']:.2f}.png"
        plt.savefig(os.path.join(SaveDir, fileName), dpi=600, bbox_inches='tight')
    
    plt.show()
    plt.ioff()
    return   

def plotGraph(P_matrix, ifSave=False, save_dir='', figsize = (15, 50)):
    nTrials, N, _ = P_matrix.shape
    ROI = np.arange(0, N)

    def plotGraph_single(P_matrix, ifSave=False, save_dir='', figsize = figsize): 
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        pcol = axs[0].pcolormesh(P_matrix[0], edgecolor='w')
        axs[0].set_xticks(ROI + 0.5, ROI + 1, fontsize=12)
        axs[0].set_yticks(ROI + 0.5, ROI + 1, fontsize=12)
        axs[0].set_ylabel("Nodes")
        axs[0].set_xlabel("Nodes")
        fig.colorbar(pcol, ax = axs[0], location='top')


        G = nx.from_numpy_array(P_matrix[0], parallel_edges=False)
        pos = nx.circular_layout(G) 

        nx.draw_networkx_nodes(G, pos, ax=axs[1], node_color='lightblue', edgecolors='darkblue', node_size=800)

        # Adding labels to the nodes
        labels = {}
        for n in range(N):
            labels[n] = f"{ROI[n] + 1:02d}"

        nx.draw_networkx_labels(G, pos, labels, ax=axs[1], font_size=8)

        all_weights = []
        #4 a. Iterate through the graph nodes to gather all the weights
        for (node1, node2, data) in G.edges(data = True):
            all_weights.append(data['weight']) 

        #4 b. Get unique weights
        unique_weights = list(set(all_weights))
        maxWeight = np.max(unique_weights)

        #4 c. Plot the edges - one by one!
        for weight in unique_weights:
            #4 d. Form a filtered list with just the weight you want to draw
            weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
            nx.draw_networkx_edges(G, pos, ax=axs[1], edge_color='darkblue', alpha = weight / maxWeight, edgelist = weighted_edges, width = 2.5)

        #Plot the graph
        axs[1].axis('off')

        hist = axs[2].hist(P_matrix[0][np.triu_indices(N, k=1)], bins=15, density=True)
        axs[2].set_xlabel("Weights")
        axs[2].set_xlim(np.min(P_matrix[0][np.triu_indices(N, k=1)]), np.max(P_matrix[0]))

        fig.suptitle(f"Network analysis with {N:02d} Nodes and {nTrials:02d} Realizations")
        plt.tight_layout()

        if ifSave:
            plt.savefig(os.path.join(save_dir, "structural_connectivity.png"), dpi=600) 
        plt.show() 

        return

    if nTrials == 1:
        plotGraph_single(P_matrix, ifSave=ifSave, save_dir=save_dir, figsize = figsize)
        return

    fig, axs = plt.subplots(nTrials, 3, figsize=(15, 50), gridspec_kw={'width_ratios':[1, 1, 1]}, layout='constrained')
    
    for q in range(nTrials):
        pcol = axs[q, 0].pcolormesh(P_matrix[q], edgecolor='w')
        axs[q, 0].set_xticks(ROI + 0.5, ROI + 1, fontsize=12)
        axs[q, 0].set_yticks(ROI + 0.5, ROI + 1, fontsize=12)
        axs[q, 0].set_ylabel("Nodes")
        axs[q, 0].set_xlabel("Nodes")
        fig.colorbar(pcol, ax = axs[q, 0], location='top')

        G = nx.from_numpy_array(P_matrix[q], parallel_edges=False)
        pos = nx.circular_layout(G) 

        nx.draw_networkx_nodes(G, pos, ax=axs[q, 1], node_color='lightblue', edgecolors='darkblue', node_size=800)

        # Adding labels to the nodes
        labels = {}
        for n in range(N):
            labels[n] = f"{ROI[n] + 1:02d}"

        nx.draw_networkx_labels(G, pos, labels, ax=axs[q, 1], font_size=8)

        all_weights = []
        #4 a. Iterate through the graph nodes to gather all the weights
        for (node1, node2, data) in G.edges(data = True):
            all_weights.append(data['weight']) 

        #4 b. Get unique weights
        unique_weights = list(set(all_weights))
        maxWeight = np.max(unique_weights)

        #4 c. Plot the edges - one by one!
        for weight in unique_weights:
            #4 d. Form a filtered list with just the weight you want to draw
            weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
            nx.draw_networkx_edges(G, pos, ax=axs[q, 1], edge_color='darkblue', alpha = weight / maxWeight, edgelist = weighted_edges, width = 2.5)

        #Plot the graph
        axs[q, 1].axis('off')

        hist = axs[q, 2].hist(P_matrix[q][np.triu_indices(N, k=1)], bins=15, density=True)
        axs[q, 2].set_xlabel("Weights")
        axs[q, 2].set_xlim(np.min(P_matrix[q][np.triu_indices(N, k=1)]), np.max(P_matrix[q]))

    fig.suptitle(f"Network analysis with {N:02d} Nodes and {nTrials:02d} Realizations")

    if ifSave:
        plt.savefig(os.path.join(save_dir, "structural_connectivity.png"), dpi=600, bbox_inches='tight') 
    plt.show() 

    return

def plot_ts(u, t_arr, t1, t2, fig_size=(10, 6), if_zoom=False, zoom_t=[], if_save=False, save_dir=None, image_name=None, **opts):
    u_plot_arr = u[:, :, t1: t2]
    t_prime = t_arr[: -t1 + t2 - 1]

    # title_str = f'Population dynamics with ' + r'$K$ = ' + f"{opts.get('Kglob'):.2f} and " + r'$I_{in}^1$ = ' + f"{opts.get('i_e1'):.3f} mV"
    # sub_title_str = r'($\sigma_e^1$ = ' + \
    #                 f"{opts.get('sig_e1'):.3f}, " + \
    #                 r'$\sigma_i^1$ = ' + \
    #                 f"{opts.get('sig_i1'):.3f}, " + \
    #                 r'$\sigma_e^2$ = ' + \
    #                 f"{opts.get('sig_e2'):.3f}, " + \
    #                 r'$\sigma_i^2$ = ' + \
    #                 f"{opts.get('sig_i2'):.3f})\n"

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

    # fig.suptitle(title_str + '\n' + sub_title_str, fontsize=24)
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
    interp = RegularGridInterpolator(fit_points, orig_response, bounds_error = False, fill_value = 0)

    eval_y_high, eval_x_high = np.meshgrid(high_res_y, high_res_x, indexing='ij')

    interp_points = np.array([eval_y_high.ravel(), eval_x_high.ravel()]).T
    response_node_01_high = interp(interp_points, method='slinear').reshape(resolution, resolution)

    response_node_01_high[response_node_01_high > 0.01] = 1
    response_node_01_high[response_node_01_high <= 0.01] = -1

    # Smoothing the response map to have continuous boundary by applying gaussian filter
    sigma = [sigma_y, sigma_x]
    smoothed_node_01_high = sp.ndimage.filters.gaussian_filter(response_node_01_high, sigma, mode='nearest')

    return high_res_y, high_res_x, smoothed_node_01_high
