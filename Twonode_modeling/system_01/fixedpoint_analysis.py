import os
import sys
import json
import time
import random
import pickle
import datetime
import numpy as np
from tqdm import tqdm

sys.path.insert(1, '../utils')

from utils import *
from plotUtils import *
from mUtils import *

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

COLOR_MAP = {
                'mediumblue': "Unstable Node",
                'darkgreen': "Stable Node",
                'red': "Unstable Saddle Point",
                'saddlebrown': "Stable Spiral",
                'purple': "Unstable Spiral",
                'black': "Circle"
            }

# Setting save dir paths
root_dir = "./results/fp/"

save_str = datetime.datetime(2024, 4, 20)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"
save_dir = os.path.join(root_dir, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)

save_file_name = os.path.join(save_dir, 'scene01_data.pkl')

# read parameters from a json file
with open('../params.json', 'r+') as f:
    params = json.load(f)

# setting grid parameters
minval = -5 / params['gamma']
maxval = 5 / params['gamma']
grid_res = 1000
n_trials = 10

# time vector
T = params['T']
time_ = np.arange(0, T)

# Initialization points
np.random.seed(0)
x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

# fixed-point calculations - nodes have single heterogeneity parameter changed
Kmin = -1; Kmax = 1; Ksteps = 10
Karr = Kmin + np.arange(0, Ksteps + 1) / Ksteps * (Kmax - Kmin)

i_min = 0; i_max = 0.5; i_steps = 500
i_values = i_min + np.arange(0, i_steps) / (i_steps - 1) * (i_max - i_min)

sig_min = 2.5; sig_max = 16.5; sig_steps = 20; delta_sig = (sig_max - sig_min) / sig_steps
eval_sigma = np.arange(2.5, 17.5, 1.25) # sig_min + np.arange(0, sig_steps) / (sig_steps - 1) * (sig_max - sig_min)

for k_in, K in enumerate(Karr):
    params['K'] = K

    for e1, sigma in enumerate(eval_sigma):
        print(f"Evaluating Sigma = {sigma}")
        params['sig_e1'] = params['sig_i1'] = params['sig_e2'] = params['sig_i2'] =  sigma

        FPvar = []
        FPs = []
        FPType = []          

        file_str_k = f"scenario_1_sigma_{params['sig_e1']:.3f}_K_{params['K']:.2f}"
        file_name = os.path.join(save_dir, file_str_k + '.pkl')

        if os.path.isfile(file_name):
            print(f"File located at:: {file_name}...Aborting!")
            
        else:
            print("Running analysis...")
            for i in tqdm(range(i_steps)):
                params['i_e1'] = i_values[i] / params['gamma'] + params['i_e2']

                fp_dict = calc_fixed_points(
                                            minval=minval,
                                            maxval=maxval,
                                            grid_res=grid_res,
                                            n_init=100,
                                            cal_grad=False,
                                            **params
                )

                eig_dict = calc_jacobian(fp_dict['fps'].T, **params)

                fps = fp_dict['fps'].T 
                fp_types = eig_dict['types']

                if len(fp_types) == 0:
                    continue

                for fp, fp_color in zip(fps, fp_types):
                    FPvar.append(i_values[i] / params['gamma'])
                    FPs.append(fp)
                    FPType.append(fp_color)

            data_dict = {'FPVar': FPvar, 'FPs': FPs, 'FPType': FPType }

            with open(file_name, 'wb') as file:
                pickle.dump(data_dict, file)