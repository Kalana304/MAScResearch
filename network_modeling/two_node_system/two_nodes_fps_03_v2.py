import os
import json
import time
import random
import pickle
import datetime
import numpy as np

from utils import *
from plot_utils import *
from measure_utils import *

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
root_dir = "./results/fp/scene03"

save_str = datetime.datetime.now()
sub_dir = f"exp-{save_str.year}-{save_str.month}-{save_str.day}"
save_dir = os.path.join(root_dir, sub_dir)

print(f"Creating dir to save results:: {save_dir}")
os.makedirs(save_dir, exist_ok = True)

save_file_name = os.path.join(save_dir, 'scene03v2_data.pkl')

data_dict = {
                "I": [],
                "K": [],
                "sig_i": [],
                "sig_e1": [],
                "sig_e2": [],
                "ue1": [],
                "ui1": [],
                "ue2": [],
                "ui2": [],
                "FP_type": []
    }

# read parameters from a json file
with open('./params.json', 'r+') as f:
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
random.seed(0)

x0_ = np.random.uniform(minval, maxval, n_trials)
x1_ = np.random.uniform(minval, maxval, n_trials)
x2_ = np.random.uniform(minval, maxval, n_trials)
x3_ = np.random.uniform(minval, maxval, n_trials)

# fixed-point calculations - nodes have single heterogeneity parameter changed
if os.path.isfile(save_file_name):
        print(f"File located at:: {save_file_name}...Aborting!")

else:
    print("Running analysis...")

    k_min = -1; k_max = 1; k_steps = 20; delta_k = (k_max - k_min) / k_steps
    i_min = -0.25; i_max = 0.25; i_steps = 10; delta_i = (i_max - i_min) / i_steps
    sige_min = 2.5; sige_max = 16.5; sige_steps = 20; delta_sige = (sige_max - sige_min) / sige_steps
    sigi_min = 2.5; sigi_max = 16.5; sigi_steps = 20; delta_sigi = (sigi_max - sigi_min) / sigi_steps

    for ie in np.arange(i_min, i_max, delta_i):
        print(f"Evaluating for I = {ie:.3f}")

        params['i_e1'] = ie / params.get('gamma')

        for k in np.arange(k_min, k_max, delta_k):
            params['K'] = k

            for sigi in np.arange(sigi_min, sigi_max, delta_sigi):
                params['sig_i1'] = params['sig_i2'] = sigi 
                
                for sige_1 in np.arange(sige_min, sige_max, delta_sige):
                    params['sig_e1'] = sige_1 

                    for sige_2 in np.arange(sige_min, sige_max, delta_sige):
                        params['sig_e2'] = sige_2

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
                            data_dict['I'].append(params.get('i_e1'))
                            data_dict['K'].append(params.get('K'))
                            data_dict['sig_e1'].append(sige_1)
                            data_dict['sig_e2'].append(sige_1)
                            data_dict['sig_i'].append(sigi)

                            data_dict['ue1'].append(fp[0]); data_dict['ui1'].append(fp[1])
                            data_dict['ue2'].append(fp[2]); data_dict['ui2'].append(fp[3])
                            data_dict['FP_type'].append(fp_color)

with open(save_file_name, 'wb') as file:
    pickle.dump(data_dict, file)


    