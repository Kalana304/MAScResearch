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
from fcMeasure import *

plt.rcParams.update({'font.size': 14})

# read parameters from a json file
with open('../params.json', 'r+') as f:
    params = json.load(f)

nTrials = 10
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
root_dir = "./results/functionality/"

save_str = datetime.datetime(2024, 5, 21)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"

save_dir = os.path.join(root_dir, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)


# Setting parameters to for the modeling
P, ROI, ROItag = modelP(nNodes, nRealizations, data_root="../data", n_files = 88,  WD="Low")
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
np.random.seed(0)
xInit = np.random.uniform(minval, maxval, (nTrials, 2 * nNodes))

Kmin = -1; Kmax = 1; Ksteps = 10
Karr = Kmin + np.arange(0, Ksteps + 1) / Ksteps * (Kmax - Kmin)

modNodes = [0]
modAmp = [0.5]
sigeArr = np.array([2.5, 7.5]); sigSteps = len(sigeArr)
sigiArr = np.array([4.4, 16.5]); sigSteps = len(sigiArr)

sigVar = np.array([0.1, 1, 2])

lowFreq = 40
midFreq = 85
highFreq = 400

window_size = 1000
overlap = 1000

for nV, _sigvar in enumerate(sigVar):
    sigCov = _sigvar * np.eye(2)

    for I in modAmp:
        params['i_mod'] = I / params['gamma']

        for nK, K in enumerate(Karr):
            params['Kglob'] = K

            for nse, _sige in enumerate(sigeArr):
                tic = time.time()
                FC_sige = {}
                fileName = f"Imod_{params['i_mod']:.3f}mV_KGlob_{params['Kglob']:.2f}_sige_{_sige:.3f}_sigVar_{_sigvar:.2f}.pkl"

                if os.path.isfile(os.path.join(save_dir, fileName)):
                    print(f"Skipping -- {fileName}")
                    continue

                for nsi, _sigi in enumerate(sigiArr):
                    FC = {
                            # 'DCOR': [],
                            # 'CORCOEF': [],
                            'PLV_LOW': [],
                            'PLV_HIGH': [],
                            # 'PLI': [],
                            # 'COHERENCE_LOW': [],
                            # 'COHERENCE_HIGH': []
                        }
                    
                    # Creating sigma array with covariance
                    params['sigma'] = np.zeros((nTrials, 2 * nNodes))

                    sigMean = np.array([_sige, _sigi])
                    for q in range(nTrials):
                        np.random.seed(q * 10 + 5)
                        sigmaNet = np.random.multivariate_normal(sigMean, sigCov, nNodes)
                        sigmaNet[modNodes, :] = sigMean

                        params['sigma'][q, :] = abs(sigmaNet.flatten())

                    # Running simulation
                    u = np.zeros((nTrials, 2 * nNodes, T))
                    u = euler_intergrate(u, xInit, modNodes = modNodes, **params)

                    plvSynchGamma = SpectrumSynch(
                                            time_series=u, 
                                            tstart=t_start, 
                                            tend=t_end, 
                                            fmin=lowFreq, 
                                            fmax=midFreq, 
                                            fs=1000, 
                                            window_size=250, 
                                            step_window=50
                                            )

                    plvSynchHFO = SpectrumSynch(
                                            time_series=u, 
                                            tstart=t_start, 
                                            tend=t_end, 
                                            fmin=midFreq, 
                                            fmax=highFreq, 
                                            fs=1000, 
                                            window_size=250, 
                                            step_window=50
                                            )
                    
                    FC["PLV_LOW"] = plvSynchGamma.calc_synchronization()
                    FC["PLV_HIGH"] = plvSynchHFO.calc_synchronization()

                    # for t in range(t_start, T - t_start, overlap):
                    #     ut = np.mean(u[:, 0:: 2, t : t + window_size], axis=0)

                    #     FC['DCOR'].append(dcor_connectivity(nNodes, ut)[0])
                    #     FC['CORCOEF'].append(ccf_connectivity(nNodes, ut)[0])
                    #     FC['PLV_HIGH'].append(plv_connectivity(nNodes, filteration(ut, midFreq, highFreq, 1000))[0])
                    #     FC['PLV_LOW'].append(plv_connectivity(nNodes, filteration(ut, lowFreq, midFreq, 1000))[0])
                    #     FC['PLI'].append(pli_connectivity(nNodes, ut)[0])
                    #     FC['COHERENCE_HIGH'].append(coh_connectivity(nNodes, ut, f_min = midFreq, f_max = highFreq, fs = 1000)[0])
                    #     FC['COHERENCE_LOW'].append(coh_connectivity(nNodes, ut, f_min = lowFreq, f_max = midFreq, fs = 1000)[0])

                    FC_sige[_sigi] = FC

                with open(os.path.join(save_dir, fileName), "wb") as _file:
                    pickle.dump(FC_sige, _file)

                toc = time.time()
                print(f"{toc - tic:.2f}s for sige = {_sige:.3f}")
