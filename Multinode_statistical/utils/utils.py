import os
import json
import time
import random
import pickle
import datetime
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

from scipy import signal
from scipy.optimize import fsolve
from sklearn.decomposition import KernelPCA

plt.rcParams.update({'font.size': 16})

# resonse function of neurons
def f(x, beta, thresh):
    return 1 / (1 + np.exp(-beta * (x - thresh)))

def F(x, beta, thresh, gamma, sig):
    vmin = -1 / gamma; vmax = 1 / gamma; nsteps = 1000
    dv = np.abs(vmax - vmin) / nsteps
    v = np.arange(vmin, vmax, dv)

    if type(x) == np.ndarray and len(x.shape) == 4:
        x = np.repeat(x[:, :, :, :, np.newaxis], nsteps, axis=4)
    sigmoid_vals = f(x + v, beta, thresh)
    normal_vals = np.exp(-v ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)

    return np.sum(dv * sigmoid_vals * normal_vals, axis=-1)

# Creating P matrix

def modelP(N, nTrials, WD="High"):
    """
        This function models the structural connectivity matrix using exponential distribution.
        Exponential distribution is used following how the weights are distributed in the 
        actual OSF data available online.

        P matrix is normalized such that \max\{\sum_{j=1}^N a_{ij}\} = 1 \forall i \in [N] and 
        sort the matrix based on the weighted degrees. The sorting will be done either from 
        highest to lowest (if modulation is applied to the highest weight degree node) or 
        from lowest to highest (if modulation is applied to the lowest weight degree node).

        parameters:
            N: (int) number of nodes
            nTrials: (int) number of trials
            WD: (str) which mode of modulation

        output:
            P: (ndarray) modeled connectivity matrices of size (nTrials x N x N)
            ROIs: (ndarray) ROI numbers from 0 to N
    """
    P = np.zeros((nTrials, N, N))
    np.random.seed(0)
    
    for q in range(nTrials):
        P[q] = np.random.standard_exponential((N, N))
        P[q] -= np.diag(np.diag(P[q]))
        P[q] = (P[q] + P[q].T) / 2

        norm_factor = np.max(P[q].sum(axis=1))
        P[q] /= norm_factor

        WeightDeg = P[q].sum(axis=1)
        DegreeInd = np.argsort(WeightDeg)

        if WD == "High":
            DegreeInd = DegreeInd[::-1]

        P[q] = P[q][:, DegreeInd][DegreeInd, :]
 
    return P, np.arange(0, N)

# Vectorized euler intergration implementation
def euler_intergrate(u, init_point, modNodes,  **opts):
    # common parameters
    beta = opts.get('beta')
    gamma = opts.get('gamma')
    theta = opts.get('theta')

    nInits, _ = init_point.shape
    nRealizations = opts.get('Abar').shape[0]
    nTrials = max(nInits, nRealizations)
    nNodes = opts.get("N")

    for q in range(nTrials):
        u[q, :, 0] = init_point[0, :] if init_point.shape[0] == 1 else  init_point[q, :]       # initializing membrane potentials
        Abar = opts.get('Abar')[0] if opts.get('Abar').shape[0] == 1 else opts.get('Abar')[q] 
        sigma = opts.get('sigma')[q, :] if len(opts.get('sigma').shape) == 2 else opts.get('sigma')
        
        for t in range(opts.get('T') - 1):
            Fvect = np.zeros((2 * nNodes, 1))
            Imod = np.zeros((2 * nNodes, 1))

            for n in range(nNodes):
                Fvect[2 * n] = F(u[q, 2 * n, t], beta, theta, gamma, sigma[2 * n])
                Fvect[2 * n + 1] = F(u[q, 2 * n + 1, t], beta, theta, gamma, sigma[2 * n + 1])

                Imod[2 * n] = opts.get('i_mod') if n in modNodes else 0

            uVect = -u[q, :, t].reshape(-1, 1) + opts.get('W') @ Fvect + opts.get('Kglob') * Abar @ u[q, :, t].reshape(-1, 1) + opts.get('i_bias') + Imod
            uVect = opts.get('dt') * opts.get('Tau') @ uVect
            u[q, :, t + 1] = u[q, :, t] + uVect.flatten()           # updating membrane potentials
            
    return u

def CoeffVariance(time_series, tstart, tend):
    """
        This function calculates the coefficient of variance computed based on following.
        [Hellyer, Peter J., et al.]
        "Local inhibitory plasticity tunes macroscopic brain dynamics and allows the emergence of functional brain networks." NeuroImage 124 (2016): 85-95.

    """
    nTrials, N, _ = time_series.shape
    N = N // 2

    CV = np.zeros(nTrials)
    MEAN = np.zeros((nTrials, N))
    STD = np.zeros((nTrials, N))

    for q in range(nTrials):
        ue_arr = time_series[q, 0::2, tstart : tend]
        ue_std = np.std(ue_arr, axis=1)
        ue_mean = np.mean(ue_arr, axis=1)
        CV[q] = np.mean(ue_std / ue_mean)
        MEAN[q, :] = ue_mean
        STD[q, :] = ue_std
    
    return CV, MEAN, STD

# Vectorized euler intergration implementation

def networked_model(u, t, _modulate,  **opts):
    # common parameters
    beta = opts.get('beta')
    gamma = opts.get('gamma')
    theta = opts.get('theta')

    nNodes = u.shape[0]
    nNodes = nNodes // 2

    Fvect = np.zeros((2 * nNodes, 1))
    Imod = np.zeros((2 * nNodes, 1))
    
    for n in range(nNodes):
        Fvect[2 * n] = F(u[2 * n], beta, theta, gamma, opts.get('sigma')[2 * n])
        Fvect[2 * n + 1] = F(u[2 * n + 1], beta, theta, gamma, opts.get('sigma')[2 * n + 1])

        Imod[2 * n] = opts.get('i_mod') if n in _modulate else 0

    u_dot = opts.get('Tau') @ (-u.reshape(-1, 1) + opts.get('W') @ Fvect + opts.get('i_bias') + Imod + opts.get('Kglob') * opts.get('Abar') @ u.reshape(-1, 1))
    u_dot = u_dot.flatten()

    return u_dot

def calc_fixed_points(
        grid_points,
        _modnodes,
        nInits,
        **opts
):
    """
        This function calculates the fixed points/steady-state points of the coupled N-node system using Powell's Hybrid Method.

        Inputs:
            grid_points: (np.ndarray) initialization grid points
            _modnodes: (list) modulating nodes in the network
            nInit: (int) no of initialization points
            opts: (dict) dictionary of parameters for the model

        Return:
            results: (dict) dictionary with fp and gradients
    """
    results = {}

    # numerical search for fps
    fps = []
    nSubs, resol = grid_points.shape

    yShuffle = deepcopy(grid_points)

    for k in range(nSubs):
        np.random.shuffle(yShuffle[k, :])

    yShuffle = np.array(yShuffle)

    init_points = random.sample(list(np.arange(0, resol)), k=nInits)
    for k in init_points:
        # find the root using Powell's method
        sol, _, ier, msg = fsolve(lambda y: networked_model(y, 0, _modnodes, **opts), yShuffle[:, k], full_output=1)

        # exclude the cases where fsolve didn't converge
        if ier == 1:
            fps.append(sol)

    fps = np.array(fps).T

    # keep the unique fps
    fps = np.array(list(set(map(tuple, np.around(fps, 4).T))))
    results = {'fps': fps.T}

    if len(results['fps']) > 0:
        _, n_fps = results['fps'].shape
    else:
        n_fps = 0

    results['n_fps'] = n_fps

    return results


# Jacobian and eigen value spectrum

def fprime(x, beta, theta):
    return beta * np.exp(-beta * (x - theta)) / ((1 + np.exp(-beta * (x - theta))) ** 2)

def Fprime(x, beta, thresh, gamma, sig):
    vmin = -1 / gamma; vmax = 1 / gamma
    nsteps = 1000
    dv = np.abs(vmax - vmin) / nsteps
    v = np.arange(vmin, vmax, dv)

    sigmoid_vals = fprime(x + v, beta, thresh)
    normal_vals = np.exp(-v ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)

    return np.sum(dv * sigmoid_vals * normal_vals, axis=-1)


def calc_jacobian(fps, **opts):
    results = {
        'eigvals': [],
        'types': []
    }

    nNodes = opts.get('N')

    # common parameters
    gamma = opts.get('gamma')
    beta = opts.get('beta')
    theta = opts.get('theta')

    for i, fp in enumerate(fps):
        Rvect = np.zeros(2 * nNodes)
        for n in range(nNodes):
            Rvect[2 * n] = Fprime(fp[2 * n], beta, theta, gamma, opts['sigma'][2 * n])
            Rvect[2 * n + 1] = Fprime(fp[2* n + 1], beta, theta, gamma, opts['sigma'][2 * n + 1])

        jacobian = opts['Tau'] @ (opts.get('Kglob') * opts['Abar'] - np.eye(2 * nNodes) + opts['W'] @ np.diag(Rvect))
        results['jacobian'] = jacobian

        # Compute and return the eigenvalues
        jacobian[np.isnan(jacobian)] = 0
        jacobian[np.isinf(abs(jacobian))] = 0

        evals = np.linalg.eigvals(jacobian)
        results['eigvals'].append(evals)

        # Analysis
        real_parts = np.real(evals).T
        img_parts = np.imag(evals).T

        if (np.round(img_parts, 1) == 0).all():
            if (real_parts >= 0).any():
                results['types'].append('red')
            elif (real_parts < 0).all():
                results['types'].append('darkgreen')

        elif (np.round(img_parts, 1) != 0).any():
            if (real_parts < 0).all():
                results['types'].append('saddlebrown')
            elif (real_parts > 0).any():
                results['types'].append('purple')
            elif (real_parts == 0).any():
                results['types'].append('black')

    return results

def dimreduction(dataDict, xVar='I', **params):
    dataD = {}
    dataD[xVar] = [x / params.get('gamma') for x in dataDict['FPVar']]

    for fps in dataDict['FPs']:
        for n in range(params.get('N')):
            key_e = f"ue{n}"
            key_i = f"ui{n}"

            if key_e not in dataD.keys():
                dataD[key_e] = [fps[2 * n]]
            else:
                dataD[key_e].append(fps[2 * n])

            if key_i not in dataD.keys():
                dataD[key_i] = [fps[2 * n + 1]]
            else:
                dataD[key_i].append(fps[2 * n + 1])
    
    colNames = list(dataD.keys())
    dataDF = pd.DataFrame.from_dict(dataD)
    dataDF = dataDF[colNames] * params.get('gamma')

    PCA_INIT = KernelPCA(n_components=1, kernel='poly')
    U = PCA_INIT.fit_transform(dataDF)

    dataD['U'] = U.flatten()
    dataD['FPType'] = dataDict['FPType']

    dataDF = pd.DataFrame.from_dict(dataD)

    return dataDF  


