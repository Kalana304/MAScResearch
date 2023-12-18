import os
import json
import time
import random
import pickle
import datetime
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from scipy import signal, linalg
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

from utils import calc_fixed_points, calc_jacobian

def spectrum_analysis(eig_vals):
    n_eigvals = len(eig_vals)
    spec_rad = np.zeros(n_eigvals)
    spec_center = np.zeros(n_eigvals, dtype=np.complex_)
    eig_real = np.zeros((n_eigvals, 4))
    eig_img = np.zeros((n_eigvals, 4))

    for ilambda, eigval in enumerate(eig_vals):
        spec_center[ilambda] = np.mean(eigval)
        spec_rad[ilambda] = abs(eigval).max()
        eig_real[ilambda, :] = eigval.real
        eig_img[ilambda, :] = eigval.imag

    if n_eigvals > 1:
        _max_rad_i = np.argmax(spec_rad)
        return spec_rad[_max_rad_i], spec_center[_max_rad_i], eig_real[_max_rad_i, :], eig_img[_max_rad_i, :]

    elif n_eigvals == 1:
        return spec_rad[0], spec_center[0], eig_real[0, :], eig_img[0, :]
    else:
        return np.inf, 0, np.inf, 0