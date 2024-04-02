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

save_str = datetime.datetime(2024, 3, 31)
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"
save_dir = os.path.join(root_dir, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)

# read parameters from a json file
with open('../params.json', 'r+') as f:
    params = json.load(f)

# setting grid parameters
minval = -5 / params['gamma']
maxval = 5 / params['gamma']
grid_res = 1000
n_trials = 10

# time vector
params['T'] = T = 10000 
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

eval_sigma_e = np.array([2.5, 7.5, 13.5]); sige_steps = len(eval_sigma_e)
eval_sigma_i = np.array([4.4, 13.5, 16.5]); sigi_steps = len(eval_sigma_i) 

sigVar = np.array([0.01, 0.1, 1, 2])

for nV, _sigvar in enumerate(sigVar):
    for k_in, K in enumerate(Karr):
        params['K'] = K

        for e1, sigma_e in enumerate(eval_sigma_e):
            for i1, sigma_i in enumerate(eval_sigma_i):
                params['sig_e1'] = sigma_e
                params['sig_i1'] = sigma_i

                sige2 = []
                sigi2 = []
                for q in range(n_trials):
                    np.random.seed(q * 10 + 5)
                    sige2.append(abs(np.random.normal(sigma_e, _sigvar, 1))[0])
                    sigi2.append(abs(np.random.normal(sigma_i, _sigvar, 1))[0])
                
                params['sig_e2'] = np.mean(np.array(sige2))
                params['sig_i2'] = np.mean(np.array(sigi2))     

                FPvar = []
                FPs = []
                FPType = []          

                file_str_k = f"K_{params['K']:.2f}_sige_{params['sig_e1']:.3f}_sigi_{params['sig_i1']:.3f}_sigVar_{_sigvar:.2f}"
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


    