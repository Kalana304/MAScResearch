import os
import json
import time
import random
import pickle
import datetime
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from tqdm import  tqdm

from utils import *
from plot_utils import *
from measure_utils import *

plt.rcParams.update({'font.size': 16})

COLOR_MAP = {
                'mediumblue': "Unstable Node",
                'darkgreen': "Stable Node",
                'red': "Unstable Saddle Point",
                'saddlebrown': "Stable Spiral",
                'purple': "Unstable Spiral",
                'black': "Circle"
            }


# read parameters from a json file
with open('./params.json', 'r+') as f:
    params = json.load(f)

# setting grid parameters
minval = -5 / params['gamma']
maxval = 5 / params['gamma']
grid_res = 100
n_trials = 5

# time vector
T = params['T'] = 10000
time_ = np.arange(0, T)

# Initialization points
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

t_start = 1000
t_end = T

# Setting save dir paths
root_dir = "./results/fp/scene02"

save_str = datetime.datetime(2024, 2, 23)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"

save_dir = os.path.join(root_dir, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)

# fixed-point projections - how system dynamics at the fixed points behaves with input value

eval_sigma_e = np.arange(2.5, 18.5, 2.5)
eval_sigma_i = np.arange(2.5, 18.5, 2.5)
eval_k_values = [-0.8, -0.2, 0.0, 0.2, 0.8]

i_min = -0.25; i_max = 0.25; i_steps = 500
i_values = i_min + np.arange(0, i_steps) / i_steps * (i_max - i_min)

for k_val in eval_k_values:
    params['K'] = k_val

    for sigma_e in eval_sigma_e:
        params['sig_e1'] = params['sig_e2'] = sigma_e
        
        for sigma_i in eval_sigma_i:
            if sigma_e == sigma_i:
                continue
            params['sig_i1'] = params['sig_i2'] = sigma_i
        
            plot_dict = {'fig_size': (10, 10), 'xlabel': r"$I^{in}_e$", 'var_name': 'external perturbation', 
                         'd_var_str': r"($\sigma_e^1 = \sigma_e^2 = $" + f"{params.get('sig_e1')}, " + r"$\sigma_i^1 = \sigma_i^2$ = " +\
                          f"{params.get('sig_i1')}, " + r"$K = $" + f"{params['K']:.2f})"
                        }
            
            fp_var = []
            fp_arr = {'ue1': [], 'ui1': [], 'ue2': [], 'ui2': []}
            fp_types = {'ue1': [], 'ui1': [], 'ue2': [], 'ui2': []}
            
            file_str_k = f"scenario_2_sigma_{params['sig_e1']:.2f}_{params['sig_i1']:.2f}_K_{params['K']:.2f}"
            file_name = os.path.join(save_dir, file_str_k + '.pkl')
            
            if os.path.isfile(file_name):
                print(f"file found at {file_name}! loading results...")
                plot_fps = pickle.load(open(file_name, 'rb'))
            
            else:
                print("File is not found! Running the analysis...")
                for i in tqdm(range(len(i_values))):
                    # print(f"Coupling Strength :: {k:.3f}")
                    params['i_e1'] = i_values[i] / params['gamma']
     
                    fp_dict = calc_fixed_points( minval = minval, maxval = maxval, grid_res = grid_res, n_init = 20, cal_grad = False, **params)
                    eig_dict = calc_jacobian(fp_dict['fps'].T, **params)
                    
                    fps = fp_dict['fps'].T   
                    fp_type = eig_dict['types']
            
                    if len(fp_type) == 0:
                        continue
                        
                    for fp, fp_color in zip(fps, fp_type):
                        fp_var.append(i_values[i])
                        fp_arr['ue1'].append(fp[0]); fp_arr['ui1'].append(fp[1]); fp_arr['ue2'].append(fp[2]); fp_arr['ui2'].append(fp[3])
                        fp_types['ue1'].append(fp_color); fp_types['ui1'].append(fp_color); fp_types['ue2'].append(fp_color); fp_types['ui2'].append(fp_color)
            
                plot_fps = {'fp_var': fp_var, 'fps': fp_arr, 'fp_types': fp_types }
                plot_fps['fp_var'] = [x / params['gamma'] for x in plot_fps['fp_var']]

                with open(file_name, 'wb') as file:
                    pickle.dump(plot_fps, file)
        
                plot_fp_projections(plot_fps_dict = plot_fps, plot_dict = plot_dict, save_dir = save_dir, if_save = True, image_name = file_str_k, **params)
