import os
import sys
import json
import time
import pickle
import datetime
import numba as nb
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

######## Creating directories ########
root_dir = "./results"
type_results = "Fu_numba"
saveStr = datetime.datetime(2025, 1, 17)
sub_dir = f"exp-{saveStr.year}-{saveStr.month}-{saveStr.day}"
save_dir = os.path.join(root_dir, type_results, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)

####### Define the parameters #######
gamma = 0.016 
dt = 0.005
beta = 4.8
theta = 0.0 
tau_e = 1 
tau_i = 0.5
wee = 100.0 
wei = 187.5 
wie = -293.75 
wii = -8.125 
i_i1 = -31.25 
i_e2 = -15.625 
i_i2 = -31.25 
Ni = 200
Ne = 800

T = 2500
T = int(T / (dt * 10))

time_ = np.arange(0, T)

t_start = int(1000 / (dt * 10))
t_end = T

# setting initial values for simulation
nTrials = 5
minval = -5 / gamma
maxval = 5 / gamma

np.random.seed(0)
x0_ = np.random.uniform(minval, maxval, nTrials)
x1_ = np.random.uniform(minval, maxval, nTrials)
x2_ = np.random.uniform(minval, maxval, nTrials)
x3_ = np.random.uniform(minval, maxval, nTrials)

initPoints = [x0_, x1_, x2_, x3_]

# Setting variable arrays
iPert = np.array([0., 0.15, 0.25, 0.35, 0.5])[::-1]     # Stimulation amplitudes

sigMin = 2.5; sigMax = 16.5; sigSteps = 25
sigValues = sigMin + np.arange(0, sigSteps) / (sigSteps - 1) * (sigMax - sigMin)    # Heterogeneity values

kMin = -1; kMax = 1; kSteps = 20
kValues = kMin + np.arange(0, kSteps) / kSteps * (kMax - kMin)  # Coupling values

@nb.njit()
def sigmoid(x):
    return 1 / (1 + np.exp(-beta * (x - theta)))

@nb.njit()
def F(x, sig):
    vmin = -1 / gamma; vmax = 1 / gamma; nsteps = 1000
    dv = np.abs(vmax - vmin) / nsteps
    v = np.arange(vmin, vmax, dv)
    normal_vals = np.exp(-v ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)

    sum = 0
    
    for i in range(nsteps):
        sum = sum +  dv * sigmoid(x + v[i]) * normal_vals[i]
    return sum

@nb.njit()
def euler_intergrate(init_points, 
                     nTrials, 
                     nNodes,
                     K,
                     sige1, sigi1, sige2, sigi2,
                     i_e1, D0=0.0                                                          
                    ):

    x0_, x1_, x2_, x3_ = init_points
    u = np.zeros((nTrials, nNodes, T))
    Fu = np.zeros((nTrials, nNodes, T))

    for k, y0 in enumerate(zip(x0_, x1_, x2_, x3_)):
        # initialize with the points
        u[k, :, 0] = y0

        for t in range(T - 1):
            u[k, 0, t + 1] = u[k, 0, t] + dt / tau_e * \
                             (-u[k, 0, t] + wee * F(u[k, 0, t], sige1) + \
                              wie * F(u[k, 1, t], sigi1) + \
                              i_e1 + K * u[k, 2, t]) + np.sqrt(2 * D0 * dt) * np.random.normal(0,1) / Ne

            u[k, 1, t + 1] = u[k, 1, t] + dt / tau_i * \
                             (-u[k, 1, t] + wei * F(u[k, 0, t], sige1) + \
                              wii * F(u[k, 1, t], sigi1) + i_i1) + np.sqrt(2 * D0 * dt) * np.random.normal(0,1) / Ni

            u[k, 2, t + 1] = u[k, 2, t] + dt / tau_e * \
                             (-u[k, 2, t] + wee * F(u[k, 2, t], sige2) + \
                              wie * F(u[k, 3, t], sigi2) + \
                              i_e2 + K * u[k, 0, t]) + np.sqrt(2 * D0 * dt) * np.random.normal(0,1) / Ne

            u[k, 3, t + 1] = u[k, 3, t] + dt / tau_i * \
                             (-u[k, 3, t] + wei * F(u[k, 2, t], sige2) + \
                              wii * F(u[k, 3, t], sigi2) + i_i2) + np.sqrt(2 * D0 * dt) * np.random.normal(0,1) / Ni
            
            Fu[k, 0, t] = F(u[k, 0, t], sige1)
            Fu[k, 1, t] = F(u[k, 1, t], sigi1)
            Fu[k, 2, t] = F(u[k, 2, t], sige2)
            Fu[k, 3, t] = F(u[k, 3, t], sigi2)
        Fu[k, 0, t + 1] = F(u[k, 0, t + 1], sige1)
        Fu[k, 1, t + 1] = F(u[k, 1, t + 1], sigi1)
        Fu[k, 2, t + 1] = F(u[k, 2, t + 1], sige2)
        Fu[k, 3, t + 1] = F(u[k, 3, t + 1], sigi2)
    return u, Fu

@nb.njit()
def _calcFu(Fu, nNodes):
    Fu_results = np.zeros((nNodes))
    
    for n in range(nNodes):
        for j in range(nTrials):
            Fu_results[n] = 1/nTrials * np.mean(Fu[j, n, t_start:])
    return Fu_results

@nb.njit()
def _simulateI( init_points, 
                nTrials, 
                nNodes,
                ie1,
                D0
            ):
    
    results = np.zeros((nNodes, kSteps, sigSteps))
    
    for k1, k in enumerate(kValues):
        for s1, sig in enumerate(sigValues):
            u, Fu = euler_intergrate(init_points = init_points, 
                        nTrials = nTrials, 
                        nNodes = nNodes,
                        K = k,
                        i_e1 = ie1,
                        sige1 = sig, sigi1 = sig, sige2 = sig, sigi2 = sig,
                        D0=D0                                                     
                        )
        
            _results = _calcFu(Fu, nNodes)
            for n in range(nNodes):
                results[n, k1, s1] = _results[n]
            
    return results

D0 = float(sys.argv[1])
D0 = D0 / (gamma * gamma)
print(f'Running simulation for D0 = {D0:.3f}')

for nI, I in enumerate(iPert):
    i_e1 = I / gamma + i_e2

    if (D0 != 0.0) and (I != 0.5):
        continue

    print(f"Evaluating for I = {I:.3f}mV")
    saveFile = f"Fu_ie_{i_e1:.3f}_D0_{D0 * (gamma * gamma):.3f}_dt_{dt}ms"

    if os.path.isfile(os.path.join(save_dir, saveFile + '.pkl')):
        print(f"file found at {os.path.join(save_dir, saveFile + '.pkl')} --> Skipping simulation!")

    else:
        tic = time.time()
        lyap_mat = _simulateI(  init_points = initPoints,
                                nTrials = nTrials,
                                nNodes = 4,
                                ie1 = i_e1,
                                D0=D0                                     
                            )
        with open(os.path.join(save_dir, saveFile + '.pkl'), 'wb') as f:
            pickle.dump(lyap_mat, f)
        
        print(f"Runtime:: {time.time() - tic:.3f}s")

    
        

