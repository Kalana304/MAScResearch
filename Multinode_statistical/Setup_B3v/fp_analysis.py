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
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

sys.path.insert(1, '../utils')

from utils import * 
from plotUtils import *
from fcMeasure import *

plt.rcParams.update({'font.size': 14})

COLOR_MAP = {
                'darkgreen': "Stable Node",
                'red': "Unstable Node",
                'saddlebrown': "Stable Spiral",
                'purple': "Unstable Spiral",
                'black': "Circle"
            }

# read parameters from a json file
with open('../params.json', 'r+') as f:
    params = json.load(f)



params["N"] = nNodes = 10
minval = -10 / params['gamma']
maxval = 10 / params['gamma']
grid_res = 1000
nTrials = 20

nRealizations = 10

# time vector
T = params['T'] = 10000
time_ = np.arange(0, T)

t_start = 1000
t_end = T

# Setting save dir paths
root_dir = "./results/fixedpoints/"

save_str = datetime.datetime(2024, 4, 4)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"

save_dir = os.path.join(root_dir, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)


# Setting parameters to for the modeling
P, ROI = modelP(N = nNodes, nTrials = nRealizations, WD = "Low")
pickle.dump(P, open(os.path.join(save_dir, "connectivity.pkl"), "wb"))

plotGraph(P, ifSave=True, save_dir=save_dir, figsize = (15, 50))

Dnet= np.zeros((nNodes, 2 * nNodes))

for n in range(nNodes):
    Dnet[n, 2 * n : 2 * n + 2] = np.array([1, 0])

# Mean connectivity matrix across 10 realizations
Abar = np.mean(P, axis=0)
Abar = (Dnet.T @ Abar) @ Dnet

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
sigeArr = np.array([2.5, 7.5, 10.5, 13.5]); sigSteps = len(sigeArr)
sigiArr = np.array([4.4, 10.5, 13.5, 16.5]); sigSteps = len(sigiArr)

i_min = 0; i_max = 0.5; i_steps = 500
i_values = i_min + np.arange(0, i_steps) / (i_steps - 1) * (i_max - i_min)

sigVar = np.array([0.01, 0.1, 1, 2])

for nV, _sigvar in enumerate(sigVar):
    sigCov = _sigvar * np.eye(2)

    for nK, K in enumerate(Karr):
        params['Kglob'] = K

        for nse, _sige in enumerate(sigeArr):
            tic = time.time()
            for nsi, _sigi in enumerate(sigiArr):
                # Creating sigma array with covariance
                sigma_arr = np.zeros((nRealizations, 2 * nNodes))

                sigMean = np.array([_sige, _sigi])
                for q in range(nRealizations):
                    np.random.seed(q * 10 + 5)
                    sigmaNet = np.random.multivariate_normal(sigMean, sigCov, nNodes)
                    sigmaNet[modNodes, :] = sigMean

                    sigma_arr[q, :] = abs(sigmaNet.flatten())

                params['sigma'] = np.mean(sigma_arr, axis=0)

                FPvar = []
                FPs = []
                FPType = []
        
                file_str_k = f"KGlob_{params['Kglob']:.2f}_sige_{params['sigma'][0]:.3f}_sigi_{params['sigma'][1]:.3f}_sigVar_{_sigvar:.2f}"
                file_name = os.path.join(save_dir, file_str_k + '.pkl')

                if os.path.isfile(file_name):
                    print('skipping evaluation!!')

                else:
                    for i in tqdm(range(i_steps)):
                        params['i_mod'] = i_values[i] / params['gamma']

                        fp_dict = calc_fixed_points(grid_points=xInit, _modnodes=modNodes, nInits=nTrials, **params)
                        eig_dict = calc_jacobian(fp_dict['fps'].T, **params)

                        fps = fp_dict['fps'].T   
                        fp_type = eig_dict['types']

                        if len(fp_type) == 0:
                            print(fps.shape)
                            continue
                            
                        for fp, fp_color in zip(fps, fp_type):
                            FPvar.append(i_values[i])
                            FPs.append(fp)
                            FPType.append(fp_color)
                            
                    plot_fps = {'FPVar': FPvar, 'FPs': FPs, 'FPType': FPType }
                
                    with open(file_name, 'wb') as file:
                        pickle.dump(plot_fps, file)

                    DF = dimreduction(plot_fps, xVar='I', **params)

                    textDelt = max(DF['U']/ params['gamma']) * 0.5
                    plt.figure(figsize=(7, 4))
                    plt.scatter(np.array(plot_fps['FPVar']) / params['gamma'], DF['U'] / params['gamma'], c=DF['FPType'], marker='.')
                    plt.ylabel(r'$\psi(\mathbf{u})$')
                    plt.text(0, max(DF['U']/ params['gamma']) + textDelt, COLOR_MAP['purple'], c='purple')
                    plt.text(23, max(DF['U']/ params['gamma']) + textDelt, COLOR_MAP['red'], c='red')
                    plt.text(18, max(DF['U']/ params['gamma']) + textDelt / 2, COLOR_MAP['black'], c='black')
                    plt.text(8, max(DF['U']/ params['gamma']) + textDelt / 2, COLOR_MAP['darkgreen'], c='darkgreen')
                    plt.text(12, max(DF['U']/ params['gamma']) + textDelt, COLOR_MAP['saddlebrown'], c='saddlebrown')
                    plt.xlim(min(np.array(plot_fps['FPVar'])) / params['gamma'], max(np.array(plot_fps['FPVar'])) / params['gamma'])
                    plt.savefig(os.path.join(save_dir, file_str_k + '.png'), dpi=600, bbox_inches='tight')