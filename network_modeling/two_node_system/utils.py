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

plt.rcParams.update({'font.size': 16})

# resonse function of neurons
def f(x, beta, thresh):
    return 1 / (1 + np.exp(-beta * (x - thresh)))

def F(x, beta, thresh, gamma, sig):
    vmin = -1 / gamma; vmax = 1 / gamma; nsteps = 1000
    dv = np.abs(vmax - vmin) / nsteps
    v = np.arange(vmin, vmax, dv)

    if type(x) == np.ndarray and len(x.shape) == 4:
        x = np.repeat(x[:, :, :, :, np.newaxis], nsteps, axis=4)
    sigmoid_vals = f(x + v, beta, thresh)
    normal_vals = np.exp(-v ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)

    return np.sum(dv * sigmoid_vals * normal_vals, axis=-1)


def coupled_node_model(x, t, **opts):
    # common parameters
    gamma = opts.get('gamma')
    beta = opts.get('beta')
    theta = opts.get('theta')
    sig_e1 = opts.get('sig_e1')
    sig_e2 = opts.get('sig_e2')
    sig_i1 = opts.get('sig_i1')
    sig_i2 = opts.get('sig_i2')

    de1_dt = (-x[0] + opts.get('wee') * F(x[0], beta, theta, gamma, sig_e1) + \
              opts.get('wie') * F(x[1], beta, theta, gamma, sig_i1) + \
              opts.get('i_e1') + opts.get('K') * x[2]) / opts.get('tau_e')

    de2_dt = (-x[2] + opts.get('wee') * F(x[2], beta, theta, gamma, sig_e2) + \
              opts.get('wie') * F(x[3], beta, theta, gamma, sig_i2) + \
              opts.get('i_e2') + opts.get('K') * x[0]) / opts.get('tau_e')

    di1_dt = (-x[1] + opts.get('wei') * F(x[0], beta, theta, gamma, sig_e1) + \
              opts.get('wii') * F(x[1], beta, theta, gamma, sig_i1) + \
              opts.get('i_i1')) / opts.get('tau_i')

    di2_dt = (-x[3] + opts.get('wei') * F(x[2], beta, theta, gamma, sig_e2) + \
              opts.get('wii') * F(x[3], beta, theta, gamma, sig_i2) + \
              opts.get('i_i2')) / opts.get('tau_i')

    return [de1_dt, di1_dt, de2_dt, di2_dt]


def euler_intergrate(u, init_point, **opts):
    # common parameters
    beta = opts.get('beta')
    gamma = opts.get('gamma')
    theta = opts.get('theta')
    x0_, x1_, x2_, x3_ = init_point

    for k, y0 in enumerate(zip(x0_, x1_, x2_, x3_)):
        # initialize with the points
        u[k, :, 0] = y0
        for t in range(opts.get('T') - 1):
            u[k, 0, t + 1] = u[k, 0, t] + opts.get('dt') / opts.get('tau_e') * \
                             (-u[k, 0, t] + opts.get('wee') * F(u[k, 0, t], beta, theta, gamma, opts.get('sig_e1')) + \
                              opts.get('wie') * F(u[k, 1, t], beta, theta, gamma, opts.get('sig_i1')) + \
                              opts.get('i_e1') + opts.get('K') * u[k, 2, t])

            u[k, 1, t + 1] = u[k, 1, t] + opts.get('dt') / opts.get('tau_i') * \
                             (-u[k, 1, t] + opts.get('wei') * F(u[k, 0, t], beta, theta, gamma, opts.get('sig_e1')) + \
                              opts.get('wii') * F(u[k, 1, t], beta, theta, gamma, opts.get('sig_i1')) + opts.get(
                                         'i_i1'))

            u[k, 2, t + 1] = u[k, 2, t] + opts.get('dt') / opts.get('tau_e') * \
                             (-u[k, 2, t] + opts.get('wee') * F(u[k, 2, t], beta, theta, gamma, opts.get('sig_e2')) + \
                              opts.get('wie') * F(u[k, 3, t], beta, theta, gamma, opts.get('sig_i2')) + \
                              opts.get('i_e2') + opts.get('K') * u[k, 0, t]
                              )

            u[k, 3, t + 1] = u[k, 3, t] + opts.get('dt') / opts.get('tau_i') * \
                             (-u[k, 3, t] + opts.get('wei') * F(u[k, 2, t], beta, theta, gamma, opts.get('sig_e2')) + \
                              opts.get('wii') * F(u[k, 3, t], beta, theta, gamma, opts.get('sig_i2')) + opts.get('i_i2')
                              )
    return u


def calc_lypunov(
        ts,
        t1,
        t2,
        init_points,
        perturb=1e-8,
        **opts
):
    orig = deepcopy(ts[:, :, t1: t2])
    ntrials, dims, _ = orig.shape
    lyap_exp = np.zeros((ntrials, dims // 2, t2 - t1 - 1))

    for k in range(ntrials):
        orig_e1 = orig[k, 0, :]
        orig_e2 = orig[k, 2, :]

        lyap_temp = np.zeros((dims // 2, t2 - t1 - 1))
        for d in range(dims):
            init_ = deepcopy(init_points)
            init_[d] += perturb
            x0_, x1_, x2_, x3_ = init_
            u_perturb = euler_intergrate(np.zeros((1, dims, opts.get('T'))), [[x0_[k]], [x1_[k]], [x2_[k]], [x3_[k]]],
                                         **opts)
            ue1_pert = u_perturb[0, 0, t1: t2]
            ue2_pert = u_perturb[0, 2, t1: t2]

            diffe1 = np.abs(ue1_pert - orig_e1)
            lyap_temp[0, :] += 1 / dims * np.log(diffe1[1:])

            diffe2 = np.abs(ue2_pert - orig_e2)
            lyap_temp[1, :] += 1 / dims * np.log(diffe2[1:])

        lyap_exp[k, 0, :] = lyap_temp[0, :]
        lyap_exp[k, 1, :] = lyap_temp[1, :]

    return lyap_exp

def calc_fixed_points(
        minval,
        maxval,
        grid_res=75,
        n_init=20,
        cal_grad=False,
        **opts
):
    """
        This function calculates the fixed points/steady-state points of the coupled 2-node system using Powell's Hybrid Method.

        Inputs:
            minval: (int) min value of the grid of points
            maxval: (int) max value of the grid of points
            grid_res: (int) no of points in the grid
            n_init: (int) no of initialization points
            cal_grad: (bool) to calculate the gradients at each point
            opts: (dict) dictionary of parameters for the model

        Return:
            results: (dict) dictionary with fp and gradients
    """
    results = {}

    # setting up grid space
    x0 = np.linspace(minval, maxval, grid_res)  # dim 1
    x1 = np.linspace(minval, maxval, grid_res)  # dim 2
    x2 = np.linspace(minval, maxval, grid_res)  # dim 3
    x3 = np.linspace(minval, maxval, grid_res)  # dim 4

    # numerical search for fps
    fps = []
    y0 = deepcopy(x0); np.random.shuffle(y0)
    y1 = deepcopy(x1); np.random.shuffle(y1)
    y2 = deepcopy(x2); np.random.shuffle(y2)
    y3 = deepcopy(x3); np.random.shuffle(y3)

    init_points = random.sample(list(np.arange(0, len(x0))), k=n_init)

    for k in init_points:
        # find the root using Powell's method
        sol, _, ier, msg = fsolve(lambda y: coupled_node_model(y, 0, **opts), [y0[k], y1[k], y2[k], y3[k]], full_output=1)

        # exclude the cases where fsolve didn't converge
        if ier == 1:
            fps.append(sol)

    fps = np.array(fps).T

    # keep the unique fps
    fps = np.array(list(set(map(tuple, np.around(fps, 4).T))))
    results = {'fps': fps.T}

    if len(results['fps']) > 0:
        _, n_fps = results['fps'].shape
    else:
        n_fps = 0

    results['n_fps'] = n_fps

    # evaluating gradients
    if cal_grad:
        print('calculating gradients for the system...')
        x0, x1, x2, x3 = np.meshgrid(x0, x1, x2, x3)  # 4-d mesh grid
        grad0, grad1, grad2, grad3 = coupled_node_model([x0, x1, x2, x3], 0, **opts)

        results['locs'] = [x0, x1, x2, x3]
        results['grads'] = [grad0, grad1, grad2, grad3]

    return results

# Jacobian and eigen value spectrum

def fprime(x, beta, theta):
    return beta * np.exp(-beta * (x - theta)) / ((1 + np.exp(-beta * (x - theta))) ** 2)

def Fprime(x, beta, thresh, gamma, sig):
    vmin = -1 / gamma; vmax = 1 / gamma;
    nsteps = 1000
    dv = np.abs(vmax - vmin) / nsteps
    v = np.arange(vmin, vmax, dv)

    if type(x) == np.ndarray and len(x.shape) == 4:
        x = np.repeat(x[:, :, :, :, np.newaxis], nsteps, axis=4)

    sigmoid_vals = fprime(x + v, beta, thresh)
    normal_vals = np.exp(-v ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)

    return np.sum(dv * sigmoid_vals * normal_vals, axis=-1)


def calc_jacobian(fps, **opts):
    results = {
        'eigvals': [],
        'types': []
    }

    # common parameters
    gamma = opts.get('gamma')
    beta = opts.get('beta')
    theta = opts.get('theta')
    sig_e1 = opts.get('sig_e1')
    sig_e2 = opts.get('sig_e2')
    sig_i1 = opts.get('sig_i1')
    sig_i2 = opts.get('sig_i2')

    # evaluate jacobian at each fixed point of system
    for i, (f_e1, f_i1, f_e2, f_i2) in enumerate(fps):
        # print(f_e1, f_i1, f_e2, f_i2)
        jacobian = np.zeros((4, 4))

        jacobian[0, 0] = (-1 + opts.get('wee') * Fprime(f_e1, beta, theta, gamma, sig_e1)) / opts.get('tau_e')
        jacobian[0, 1] = opts.get('wie') * Fprime(f_i1, beta, theta, gamma, sig_i1) / opts.get('tau_e')
        jacobian[0, 2] = opts['K'] / opts.get('tau_e')
        jacobian[0, 3] = 0

        jacobian[1, 0] = opts.get('wei') * Fprime(f_e1, beta, theta, gamma, sig_e1) / opts.get('tau_i')
        jacobian[1, 1] = (-1 + opts.get('wii') * Fprime(f_i1, beta, theta, gamma, sig_i1)) / opts.get('tau_i')
        jacobian[1, 2] = 0
        jacobian[1, 3] = 0

        jacobian[2, 0] = opts['K'] / opts.get('tau_e')
        jacobian[2, 1] = 0
        jacobian[2, 2] = (-1 + opts.get('wee') * Fprime(f_e2, beta, theta, gamma, sig_e2)) / opts.get('tau_e')
        jacobian[2, 3] = opts.get('wie') * Fprime(f_i2, beta, theta, gamma, sig_i2) / opts.get('tau_e')

        jacobian[3, 0] = 0
        jacobian[3, 1] = 0
        jacobian[3, 2] = opts.get('wei') * Fprime(f_e2, beta, theta, gamma, sig_e2) / opts.get('tau_i')
        jacobian[3, 3] = (-1 + opts.get('wii') * Fprime(f_i2, beta, theta, gamma, sig_i2)) / opts.get('tau_i')

        results['jacobian'] = jacobian

        # Compute and return the eigenvalues
        if np.isnan(sum(jacobian.flatten())) or np.isinf(sum(jacobian.flatten())):
            return results

        evals = np.linalg.eigvals(jacobian)
        results['eigvals'].append(evals)

        # Analysis
        real_parts = np.real(evals).T
        img_parts = np.imag(evals).T

        if (img_parts == 0).all():
            if (real_parts > 0).all():
                results['types'].append('mediumblue')
            elif (real_parts < 0).all():
                results['types'].append('darkgreen')
            else:
                results['types'].append('red')

        elif (img_parts != 0).any():
            if (real_parts < 0).all():
                results['types'].append('saddlebrown')
            elif (real_parts > 0).any():
                results['types'].append('purple')
            elif (real_parts == 0).any():
                results['types'].append('black')

    return results