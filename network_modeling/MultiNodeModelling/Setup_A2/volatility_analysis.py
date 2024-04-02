import os
import sys
import glob
import time
import json
import pickle
import random
import datetime
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

import networkx as nx
import scipy as sp

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

sys.path.insert(1, '../utils')

from utils import * 
from plotUtils import *

plt.rcParams.update({'font.size': 14})

# read parameters from a json file
with open('../params.json', 'r+') as f:
    params = json.load(f)

nTrials = 1
nRealizations = 10

params["N"] = nNodes = 10
minval = -5 / params['gamma']
maxval = 5 / params['gamma']

# time vector
T = params['T'] = 10000
time_ = np.arange(0, T)

t_start = 1000
t_end = T

# Setting save dir paths
root_dir = "./results/volatility/"

save_str = datetime.datetime(2024, 3, 30)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"

save_dir = os.path.join(root_dir, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)


# Setting parameters to for the modeling
P, ROI = modelP(N = nNodes, nTrials = nRealizations, WD = "High")
pickle.dump(P, open(os.path.join(save_dir, "connectivity.pkl"), "wb"))

plotGraph(P, ifSave=True, save_dir=save_dir, figsize = (15, 50))

Dnet= np.zeros((nNodes, 2 * nNodes))

for n in range(nNodes):
    Dnet[n, 2 * n : 2 * n + 2] = np.array([1, 0])

Abar = np.zeros((nRealizations, 2 * nNodes, 2 * nNodes))
for q in range(nRealizations):
    Abar[q] = (Dnet.T @ P[q]) @ Dnet

params["Abar"] = Abar 

tau = np.diag(np.array([1 / params['tau_e'], 1 / params['tau_i']]))
params['Tau'] = Tau = np.kron(np.eye(nNodes), tau)

w = np.array([[params['wee'], params['wie']], [params['wei'], params['wii']]])
params['W'] = W = np.kron(np.eye(nNodes), w)

ibias = np.array([params['i_ebias'], params['i_ibias']])
params['i_bias'] = i_bias = np.repeat(ibias[np.newaxis,...], nNodes, axis=0).flatten().reshape(-1, 1)

# Initialization
random.seed(0)
xInit = np.random.uniform(minval, maxval, (nTrials, 2 * nNodes))

Kmin = -1; Kmax = 1; Ksteps = 10
Karr = Kmin + np.arange(0, Ksteps + 1) / Ksteps * (Kmax - Kmin)

modNodes = [0]
modAmp = [0.5, 0.35, 0.25, 0.15, 0.0]
sigArr = np.array([2.5, 4.4, 7.5, 10.5, 13.5, 16.5]); sigSteps = len(sigArr)

for I in modAmp:
    params['i_mod'] = I / params['gamma']

    for nK, K in enumerate(Karr):
        params['Kglob'] = K

        _lyapunov = {
                        "mod_node": np.zeros((sigSteps, sigSteps)),
                        "mod_node_var": np.zeros((sigSteps, sigSteps)),
                        "nonmod_nodes": np.zeros((sigSteps, sigSteps)),
                        "nonmod_nodes_var": np.zeros((sigSteps, sigSteps)),
                        "network": np.zeros((sigSteps, sigSteps))
                    }
        _cv = {
                "mean_cv": np.zeros((sigSteps, sigSteps)),
                "var_cv": np.zeros((sigSteps, sigSteps))
        }
        
        fileName = f"Imod_{params['i_mod']:.3f}mV_KGlob_{params['Kglob']:.2f}"

        for nse, _sige in enumerate(sigArr):
            tic = time.time()
            for nsi, _sigi in enumerate(sigArr):
                
                params['sigma'] = np.ones(2 * nNodes)
                params['sigma'][0::2] = _sige
                params['sigma'][1::2] = _sigi 

                # Running simulation
                u = np.zeros((nRealizations, 2 * nNodes, T))
                u = euler_intergrate(u, xInit, modNodes = modNodes, **params)

                _lyap = np.zeros((nRealizations, nNodes, T - t_start - 1))
                for j in range(nRealizations):
                    _diff_mod = np.abs(np.diff(u[j, 0::2, t_start:], axis=1))
                    _lyap[j, :, :] = np.log(_diff_mod)

                _meanlyap = np.mean(_lyap, axis=2)

                _lyapunov['mod_node'][nse, nsi] = np.mean(_meanlyap[:, 0])
                _lyapunov['mod_node_var'][nse, nsi] = np.std(_meanlyap[:, 0])
                _lyapunov['nonmod_nodes'][nse, nsi] = np.mean(_lyap[:, 1:])
                _lyapunov['nonmod_nodes_var'][nse, nsi] = np.std(_lyap[:, 1:])
                _lyapunov['network'][nse, nsi] = np.mean(_lyap)
                
                _cvmeasure = CoeffVariance(u, t_start, t_end)
                _cv["mean_cv"][nse, nsi] = np.mean(_cvmeasure)
                _cv["var_cv"][nse, nsi] = np.std(_cvmeasure)

            toc = time.time()
            print(f"{toc - tic:.2f}s for sige = {_sige:.3f}")
        
        with open(os.path.join(save_dir, fileName + "_lyap.pkl"), "wb") as _file:
            pickle.dump(_lyapunov, _file)

        with open(os.path.join(save_dir, fileName + "_cv.pkl"), "wb") as _file:
            pickle.dump(_cv, _file)
