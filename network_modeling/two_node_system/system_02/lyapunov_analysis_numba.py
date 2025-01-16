import os
import sys
import json
import time
import pickle
import datetime
import argparse
import numba as nb
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

######## Creating directories ########
root_dir = "./results"
type_results = "lyapunov_numba"
saveStr = datetime.datetime(2025, 1, 15)
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

sigeMin = 2.5; sigeMax = 16.5; sigeSteps = 20
sigeValues = sigeMin + np.arange(0, sigeSteps) / (sigeSteps - 1) * (sigeMax - sigeMin)    # Heterogeneity values

sigiMin = 2.5; sigiMax = 16.5; sigiSteps = 20
sigiValues = sigiMin + np.arange(0, sigiSteps) / (sigiSteps - 1) * (sigiMax - sigiMin)    # Heterogeneity values

kMin = -1; kMax = 1; kSteps = 10
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
                     i_e1,
                     dt, T                                                          
                    ):

    x0_, x1_, x2_, x3_ = init_points
    u = np.zeros((nTrials, nNodes, T))

    for k, y0 in enumerate(zip(x0_, x1_, x2_, x3_)):
        # initialize with the points
        u[k, :, 0] = y0

        for t in range(T - 1):
            u[k, 0, t + 1] = u[k, 0, t] + dt / tau_e * \
                             (-u[k, 0, t] + wee * F(u[k, 0, t], sige1) + \
                              wie * F(u[k, 1, t], sigi1) + \
                              i_e1 + K * u[k, 2, t])

            u[k, 1, t + 1] = u[k, 1, t] + dt / tau_i * \
                             (-u[k, 1, t] + wei * F(u[k, 0, t], sige1) + \
                              wii * F(u[k, 1, t], sigi1) + i_i1)

            u[k, 2, t + 1] = u[k, 2, t] + dt / tau_e * \
                             (-u[k, 2, t] + wee * F(u[k, 2, t], sige2) + \
                              wie * F(u[k, 3, t], sigi2) + \
                              i_e2 + K * u[k, 0, t])

            u[k, 3, t + 1] = u[k, 3, t] + dt / tau_i * \
                             (-u[k, 3, t] + wei * F(u[k, 2, t], sige2) + \
                              wii * F(u[k, 3, t], sigi2) + i_i2)
    return u

@nb.njit()
def _calcLyapunov(u, t_start, T, dt):
    lyapunov_1 = np.zeros((T - t_start - 1))
    for j in range(nTrials):
        diff_1 = np.abs(np.diff(u[j, 0, t_start:])) / (dt * 10)
        lyapunov_1 += 1 / nTrials * np.log(diff_1)
    
    return np.mean(lyapunov_1)

@nb.njit()
def _simulateIK( init_points, 
                nTrials, 
                nNodes,
                K,
                ie1,
                T, dt, t_start
            ):
    
    results = np.zeros((sigeSteps, sigiSteps))
    
    for e1, sigE in enumerate(sigeValues):
        for e2, sigI in enumerate(sigiValues):
            u = euler_intergrate(init_points = init_points, 
                    nTrials = nTrials, 
                    nNodes = nNodes,
                    K = K,
                    i_e1 = ie1,
                    sige1 = sigE, sigi1 = sigI, sige2 = sigE, sigi2 = sigI,
                    dt=dt, T=T                                                          
                    )
            
            lyapVal = _calcLyapunov(u=u, dt=dt, T=T, t_start=t_start)
            results[e1, e2] = lyapVal

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System02_Lyapunov")
    parser.add_argument('-I', help='Enter Ie_1 value to simulate', type=float, choices=iPert)
    parser.add_argument('-dt', help='Enter dt value to simulate in ms', type=float)

    args = parser.parse_args()

    I = args.I / gamma + i_e2
    dt = args.dt / 10

    T = 2000
    T = int(T / (dt * 10))

    time_ = np.arange(0, T)

    t_start = int(1000 / (dt * 10))
    t_end = T

    print(f"Running simulation with dt = {dt * 10}ms for I = {I:.3f}mV...")

    for k1, K in enumerate(kValues):
        saveFile = f"lyap_K_{K:.2f}_ie_{I:.3f}"

        if os.path.isfile(os.path.join(save_dir, saveFile + '.pkl')):
            print(f"file found at {os.path.join(save_dir, saveFile + '.pkl')} --> Skipping simulation!")

        else:
            tic = time.time()
            print("Starting Simulation....")
            lyap_mat = _simulateIK( init_points = initPoints,
                                    nTrials = nTrials,
                                    nNodes = 4,
                                    ie1 = I,
                                    K = K, 
                                    T = T,
                                    dt=dt, 
                                    t_start=t_start                                     
                                )
            print("Finished Simulation!")
            with open(os.path.join(save_dir, saveFile + '.pkl'), 'wb') as f:
                pickle.dump(lyap_mat, f)
        
            print(f"Runtime (K = {K:.2f}):: {time.time() - tic:.3f}s")

    
        

