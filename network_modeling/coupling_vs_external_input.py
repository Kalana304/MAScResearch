###########################################################################################################################
#
# This code analyzes how the nature of fixed points of the two-node coupled system changes with the coupling coefficient
# and the external input applied on Node 01 while keeping the firing rate heterogeneity parameters constant. The results 
# will be plotted on a 3D graph.
#
# Author: Kalana Abeywardena
# Created on: 23rd October 2023
#
##########################################################################################################################

import os
import json
import time
import pickle
import datetime
import numpy as np

from utility import _fixed_point_calc, _jacobian_calc, plot_3d_fixed_points_K_vs_i

# Define parameters for the coupled system

parameters = {}

parameters['T'] = T = 2500            # No. of time points
parameters['dt'] = dt = 0.1            # Integration time step
time_arr = np.arange(0, T)

parameters['gamma'] = gamma = 0.016       # Conversion parameter to mV
parameters['wee'] = wee = 1.6 / gamma   # E -> E subpopulation synaptic weight  
parameters['wei'] = wei = 3 / gamma     # I -> E subpopulation synaptic weight
parameters['wie'] = wie = -4.7 / gamma  # E -> I subpopulation synaptic weight
parameters['wii'] = wii = -0.13 / gamma # I -> I subpopulation synaptic weight
parameters['tau_e'] = tau_e = 1           # Time scale for E subpopulation
parameters['tau_i'] = tau_i = 1 / 2       # Time scale for I subpopulation

parameters['beta'] = beta = 300 * gamma  # slope of firing function

parameters['ie'] = e_bias = -0.25 / gamma  # Bias for E subpopulation
parameters['ii'] = i_bias = -0.5 / gamma   # Bias for I subpopulation

parameters['D'] = D = 0.000 / (gamma ** 2)
parameters['N'] = N = 2

# heterogeneity parameters for excitatory population
sigma_e = [
            [2.5, 7.8], [2.5, 2.5], [2.5, 2.5], [2.5, 7.8],
            [7.8, 7.8], [7.8, 2.5], [7.8, 2.5], [7.8, 7.8],
            [2.5, 7.8], [2.5, 2.5], [2.5, 2.5], [2.5, 7.8],
            [7.8, 7.8], [7.8, 2.5], [7.8, 2.5], [7.8, 7.8],
        ]

# heterogeneity parameters for inhibitory population
sigma_i = [
            [4.4, 16.25], [4.4, 4.4], [4.4, 16.25], [4.4, 4.4],
            [16.25, 16.25], [16.25, 4.4], [16.25, 16.25], [16.25, 4.4],
            [16.25, 16.25], [16.25, 4.4], [16.25, 16.25], [16.25, 4.4],
            [4.4, 16.25], [4.4, 4.4], [4.4, 16.25], [4.4, 4.4]
        ]       

# Define save parameters
log_time = datetime.datetime.now()

results_dir = 'results'
sub_dir = f"log_{log_time.year}_{log_time.month}_{log_time.day}_{log_time.strftime('%H')}_{log_time.strftime('%M')}"
os.makedirs(os.path.join(results_dir, sub_dir), exist_ok = True)


# Creating coupling array
K_min = -1; K_max = 1; Ksteps = 50
K_range = K_min + np.arange(0, Ksteps) / Ksteps * (K_max - K_min)

# setting the input
Imin = -0.25 / gamma; Imax = 0.25 / gamma; Isteps = 50
I_range = Imin + np.arange(0, Isteps) / Isteps * (Imax - Imin)

param_file_name = "two_node_system_params.json"

simulation_no = 1

for sigma_e_, sigma_i_ in zip(sigma_e, sigma_i):
    print(f"\n\nRunning simulations {simulation_no} / {len(sigma_e)}")

    parameters['sigma_e'] = sigma_e_
    parameters['sigma_i'] = sigma_i_
    
    write_file_name = f"sig_e1_{sigma_e_[0]:.3f}_sig_i1_{sigma_i_[0]:.3f}_sig_e2_{sigma_e_[1]:.3f}_sig_i2_{sigma_i_[1]:.3f}_3D.pkl"

    write_dict = {
                    'fixed_points': [],
                    'point_type': []
                }

    for niter, K in enumerate(K_range):
        if niter % 100 == 0:
            print(f"K :: {K :.3f}")

        fixedTrack = []
        fixedColors = []

        parameters['K'] = K    # setting the couping parameter

        tic = time.time()
        for I in range(Isteps):
            parameters['it'] = [I_range[I], 0]          # external input on node 1 and node 2

            fp_dict = _fixed_point_calc(
                                        minval = -1, 
                                        maxval = 1, 
                                        resolution = 75, 
                                        nsamples = 80, 
                                        cal_grad = False, 
                                        **parameters
                                        )
            fp_analysis_dict = _jacobian_calc(fp_dict['fixed'].T, **parameters)

            fixedTrack.append(fp_dict['fixed'].T)
            fixedColors.append(fp_analysis_dict['types'])

        write_dict['fixed_points'].append(fixedTrack)
        write_dict['point_type'].append(fixedColors)
        
        toc = time.time()
        print(f"Simulation End! Time taken = {toc - tic :.3f}s")

    with open(os.path.join(results_dir, sub_dir, write_file_name), 'wb') as file:
        pickle.dump(write_dict, file)

    plot_3d_fixed_points_K_vs_i(
                                i_range = I_range, 
                                k_range = K_range, 
                                fps = write_dict['fixed_points'], 
                                colors = write_dict['point_type'], 
                                save_path = os.path.join(results_dir, sub_dir), 
                                fig_size = (18, 18), 
                                **parameters
                                )
    
    simulation_no += 1

parameters['sigma_e'] = sigma_e
parameters['sigma_i'] = sigma_i

with open(os.path.join(results_dir, sub_dir, param_file_name), 'w+') as file:
        json.dump(parameters, file, indent = 4)
                            