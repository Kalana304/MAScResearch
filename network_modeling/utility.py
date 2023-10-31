import os
import random
import numpy as np
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

plt.style.use('seaborn-bright')
plt.rcParams.update({'font.size': 20})

COLOR_MAP = {
                'mediumblue': "Unstable Node",
                'darkgreen': "Stable Node",
                'red': "Unstable Saddle Point",
                'saddlebrown': "Stable Spiral",
                'purple': "Unstable Spiral",
                'black': "Circle"
            }

                
# Firing functions (mean-field model)
def f(x, beta, theta):
    return 1 / (1 + np.exp(-beta * (x - theta)))

def F(x, beta, thresh, gamma, sig):
    vmin = -1 / gamma; vmax = 1 / gamma; nsteps = 1000
    dv = np.abs(vmax - vmin) / nsteps
    v = np.arange(vmin, vmax, dv)

    if type(x) == np.ndarray and len(x.shape) == 4:
        x = np.repeat(x[:, :, :, :, np.newaxis], nsteps, axis=4)
    sigmoid_vals = f(x + v, beta, thresh)
    normal_vals = np.exp(-v ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)

    return np.sum(dv * sigmoid_vals * normal_vals, axis=-1)

def fprime(x, beta, theta):
    return beta * np.exp(-beta * (x - theta)) / ((1 + np.exp(-beta * (x - theta))) ** 2)

def Fprime(x, beta, thresh, gamma, sig):
    vmin = -1 / gamma; vmax = 1 / gamma
    nsteps = 1000
    dv = np.abs(vmax - vmin) / nsteps
    v = np.arange(vmin, vmax, dv)

    if type(x) == np.ndarray and len(x.shape) == 4:
        x = np.repeat(x[:, :, :, :, np.newaxis], nsteps, axis=4)

    sigmoid_vals = fprime(x + v, beta, 0.0)
    normal_vals = np.exp(-v ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)

    return np.sum(dv * sigmoid_vals * normal_vals, axis=-1)

# mean-field model 
def model(x, **opts):
    wee = opts.get('wee')
    wei = opts.get('wei')
    wie = opts.get('wie')
    wii = opts.get('wii')
    
    ie = opts.get('ie')
    ii = opts.get('ii')
    it = opts.get('it')
    K = opts.get('K')

    beta = opts.get('beta')
    gamma = opts.get('gamma')
    sigma_e = opts.get('sigma_e')
    sigma_i = opts.get('sigma_i')

    grad_vect = []
 
    for i in range(2):
        _sum_e = 0
        _sum_i = 0
        
        k = i * 2
        _sum_e += -x[k] + wee * F(x[k], beta, 0.0, gamma, sigma_e[i]) + wie * F(x[k + 1], beta, 0.0, gamma, sigma_i[i]) + ie
        _sum_e += K * x[(k + 2) // 2]

        _sum_e += it[i]
        _sum_e /= opts.get('tau_e')

        _sum_i += -x[k + 1] + wei * F(x[k], beta, 0.0, gamma, sigma_e[i]) + wii * F(x[k + 1], beta, 0.0, gamma, sigma_i[i]) + ii
        _sum_i /= opts.get('tau_i')

        grad_vect += [_sum_e, _sum_i]
    return grad_vect

def _fixed_point_calc(minval, maxval, resolution = 75, nsamples = 20, cal_grad = False, **opts):
    '''
        This fucntion numerically calculates the possible fixed point(s) of the neuronal dynamic system using 
        a non-linear root finding algorithm within a 4D grid.
        
        Inputs:
            minval:     (float) min. value of the grid dimension
            maxval:     (float) max. value of the grid dimension
            resoution:  (int) no. of points in the grid to evaluate
            nsmaples:   (int) no. of points to initiate trajectories
            cal_grad:   (bool) whether to calculate the gradients on the grid points  

        Output:
            return_dict: (dict) contains gradients (if cal_grad = True) and fixed points

    '''
    # Set up the 4-D space
    x0 = np.linspace(minval, maxval, resolution)
    x1 = np.linspace(minval, maxval, resolution)
    x2 = np.linspace(minval, maxval, resolution)
    x3 = np.linspace(minval, maxval, resolution)

    x0, x1, x2, x3 = np.meshgrid(x0, x1, x2, x3)

    # Evaluate the gradients
    if cal_grad:
        print(f'Calculating gradients over {resolution} x {resolution} x {resolution} x {resolution} grid!')
        grad0, grad1, grad2, grad3 = model([x0, x1, x2, x3], **opts)
    else:
        grad0, grad1, grad2, grad3 = None, None, None, None

    # Numerical search for fixed points
    fixed_points = []
    y0 = x0.ravel()
    y1 = x1.ravel()
    y2 = x2.ravel()
    y3 = x3.ravel()

    inits = random.sample(list(np.arange(0, resolution ** 4)), k = nsamples)

    for k in inits:
        sol, _, ier, msg = fsolve(lambda y: model(y, **opts), [y0[k], y1[k], y2[k], y3[k]], full_output=1)
        if ier == 1:                            # Exclude the cases where fsolve didn't converge
            fixed_points.append(sol)

    fixed_points = np.array(fixed_points).T
    fixed_points = np.array(list(set(map(tuple, np.around(fixed_points, 4).T))))   # Unique fixed points

    return_dict = {
                    'grads': [grad0, grad1, grad2, grad3],
                    'fixed': fixed_points.T
                }
    return return_dict

def _plot_traj(minval, maxval, resolution, 
               init_traj, traj_series, 
               fixed_nsample, cal_grad = False, 
               plot_dim = None, fig_size = (5, 5), save_path = None, **opts):
    
    assert type(plot_dim) == list and len(plot_dim) == 2, "Wrong dims to plot"

    dim_labels = [r'$E_1$', r'$I_1$', r'$E_2$', r'$I_2$']

    # Set up the 4-D space
    x0 = np.linspace(minval, maxval, resolution)
    x1 = np.linspace(minval, maxval, resolution)
    x2 = np.linspace(minval, maxval, resolution)
    x3 = np.linspace(minval, maxval, resolution)

    x0, x1, x2, x3 = np.meshgrid(x0, x1, x2, x3)
    grid = [x0, x1, x2, x3]

    output_dict = _fixed_point_calc(
                                    minval = minval, 
                                    maxval = maxval, 
                                    resolution = resolution, 
                                    nsamples = fixed_nsample, 
                                    cal_grad = cal_grad,  
                                    **opts
                                    )
    
    _dim0, _dim1 = plot_dim
    dim0, dim1 = grid[_dim0], grid[_dim1]
    grad0, grad1 = output_dict['grads'][_dim0], output_dict.get('grads')[_dim1]
    fp = output_dict['fixed']

    if (_dim0 == 0 or _dim0 == 1):
        dim0 = dim0[:, :, 0, 0]; grad0 = grad0[:, :, 0, 0]

    elif (_dim0 == 2 or _dim0 == 3):
        dim0 = dim0[0, 0, :, :]; grad0 = grad0[0, 0, :, :]

    if (_dim1 == 0 or _dim1 == 1):
        dim1 = dim1[:, :, 0, 0]; grad1 = grad1[:, :, 0, 0]

    elif (_dim1 == 2 or _dim1 == 3):
        dim1 = dim1[0, 0, :, :]; grad1 = grad1[0, 0, :, :]

    xlabel = dim_labels[_dim1]
    ylabel = dim_labels[_dim0]

    # Plotting the vector field in the state space (E, I)
    plt.figure(figsize = fig_size)
    plt.quiver(dim1, dim0, grad1, grad0, pivot='mid', alpha=.8)
    plt.xlim([minval, maxval]); plt.ylim([minval, maxval])
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid()

    ninits, sys_dim, _ = traj_series.shape

    x0_, x1_, x2_, x3_ = init_traj

    for k, y0 in enumerate(zip(x0_, x1_, x2_, x3_)):
        xSolve = traj_series[k, :, :]
        xdim0, xdim1 = xSolve[plot_dim]
        # Plot the solution in the state space
        plt.plot(xdim1, xdim0, '-', )

        # Plot the starting point
        plt.scatter(y0[_dim1], y0[_dim0], marker='*', c='r', s=300, label=f"{ylabel} = {y0[_dim0]:.3f} {xlabel} = {y0[_dim1]:.3f}")

    # Plot the fixed points identified
    plt.scatter(fp[_dim1], fp[_dim0], marker='o', c='k', s=100, label="Stationary points")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncols=2, fancybox=True, shadow=True)
    plt.show()
    return

def _plot_timeseries(x, t, sigmas, figsize = (10, 6), save_path = None):
    # Plot the solution in time
    plt.figure(figsize = figsize)

    dim_labels = [r'$E_1$', r'$I_1$', r'$E_2$', r'$I_2$']
    t0 = t[0]
    t1 = t[-1]

    for i in range(len(dim_labels) // 2):
        plt.subplot(4, 1, 2 * i + 1)
        plt.plot(t, np.mean(x[:, 2 * i, :], axis=0), '-', label="excitatory")

        plt.ylabel(dim_labels[2 * i]); plt.xlabel(r'$t$')
        plt.xlim([t0, t1])
        
        plt.title(f'Population {i + 1:1d} :: Excitatory Dynamics' + r' ($\sigma_e$ = ' + f'{sigmas[2 * i]})')

        plt.subplot(4, 1, 2 * i + 2)
        plt.plot(t, np.mean(x[:, 2 * i + 1, :], axis=0), '-', label="inhibitory")

        plt.ylabel(dim_labels[2 * i + 1]); plt.xlabel(r'$t$')
        plt.xlim([t0, t1])
        plt.title(f'Population {i + 1:1d} :: Inhibitory Dynamics' + r' ($\sigma_i$ = ' + f'{sigmas[2 * i + 1]})')

    plt.tight_layout()
    plt.show()
    return

def _jacobian_calc(fixed_points, **opts):
    analysis_results = {
                        'eigvals': [],
                        'types': []
                        }

    wee = opts.get('wee')
    wei = opts.get('wei')
    wie = opts.get('wie')
    wii = opts.get('wii')
    
    ie = opts.get('ie')
    ii = opts.get('ii')
    it = opts.get('it')
    K = opts.get('K')

    beta = opts.get('beta')
    gamma = opts.get('gamma')
    sigma_e = opts.get('sigma_e')
    sigma_i = opts.get('sigma_i')
    gamma = opts.get('gamma')

    # Jacobian Analysis
    for i, (fE1, fI1, fE2, fI2) in enumerate(fixed_points):
        J = np.zeros((4, 4))

        J[0,0] = (1 / opts.get('tau_e')) * (-1 + wee * Fprime(fE1, beta, 0.0, gamma, sigma_e[0]))
        J[0,1] = (1 / opts.get('tau_e')) * wie * Fprime(fI1, beta, 0.0, gamma, sigma_i[0])
        J[0,2] = (1 / opts.get('tau_e')) * opts['K']
        J[0,3] = 0

        J[1,0] = (1 / opts.get('tau_i')) * wei * Fprime(fE1, beta, 0.0, gamma, sigma_e[0])
        J[1,1] = (1 / opts.get('tau_i')) * (-1 + wii * Fprime(fI1, beta, 0.0, gamma, sigma_i[0]))
        J[1,2] = 0
        J[1,3] = 0

        J[2,0] = (1 / opts.get('tau_e')) * opts['K']
        J[2,1] = 0
        J[2,2] = (1 / opts.get('tau_e')) * (-1 + wee * Fprime(fE2, beta, 0.0, gamma, sigma_e[1]))
        J[2,3] = (1 / opts.get('tau_e')) * opts['wie'] / gamma * Fprime(fI2, beta, 0.0, gamma, sigma_i[1])

        J[3,0] = 0
        J[3,1] = 0
        J[3,2] = (1 / opts.get('tau_i')) * wei * Fprime(fE2, beta, 0.0, gamma, sigma_e[1])
        J[3,3] = (1 / opts.get('tau_i')) * (-1 + wii * Fprime(fI2, beta, 0.0, gamma, sigma_i[1]))

        # Compute and return the eigenvalues
        if np.isnan(sum(J.flatten())) or np.isinf(sum(J.flatten())):
            return analysis_results

        evals = np.linalg.eig(J)[0]
        analysis_results['eigvals'].append(evals)

        # Analysis
        real_parts = np.real(evals).T
        img_parts = np.imag(evals).T

        if (img_parts == 0).all():
            if (real_parts > 0).all():
                analysis_results['types'].append('mediumblue')

            elif (real_parts < 0).all():
                analysis_results['types'].append('darkgreen')

            else:
                analysis_results['types'].append('red')

        elif (img_parts != 0).any():
            if (real_parts < 0).all():
                analysis_results['types'].append('saddlebrown')
 
            elif (real_parts > 0).any():
                analysis_results['types'].append('purple')

            elif (real_parts == 0).any():
                analysis_results['types'].append('black')

    return analysis_results

# Plotting 3D data
def plot_3d_fixed_points_K_vs_i(i_range, k_range, fps, colors, save_path, fig_size = (18, 18), **opts):
    fig = plt.figure(figsize = fig_size)
    ax1 = fig.add_subplot(2, 2, 1, projection = '3d')
    ax2 = fig.add_subplot(2, 2, 2, projection = '3d')
    ax3 = fig.add_subplot(2, 2, 3, projection = '3d')
    ax4 = fig.add_subplot(2, 2, 4, projection = '3d')

    for k in range(len(k_range)):
        for i in range(len(i_range)):
            fixed_point = fps[k][i]
            fixed_color = colors[k][i]

            for fp, color in zip(fixed_point, fixed_color):
                sct1 = ax1.scatter3D(k_range[k], i_range[i], fp[0], marker='.', c = color)
                sct2 = ax2.scatter3D(k_range[k], i_range[i], fp[1], marker='.', c = color)
                sct3 = ax3.scatter3D(k_range[k], i_range[i], fp[2], marker='.', c = color)
                sct4 = ax4.scatter3D(k_range[k], i_range[i], fp[3], marker='.', c = color)

    ax1.set_xlabel(r'$K_1$'); ax1.set_ylabel(r'$I^1_{in}$'); ax1.set_zlabel(r'$U_e^1$')
    ax1.set_title(f"Population 01 - Excitatory Dynamics")

    ax2.set_xlabel(r'$K_1$'); ax2.set_ylabel(r'$I^1_{in}$'); ax2.set_zlabel(r'$U_i^1$')
    ax2.set_title(f"Population 01 - Inhibitory Dynamics")

    ax3.set_xlabel(r'$K_1$'); ax3.set_ylabel(r'$I^1_{in}$'); ax3.set_zlabel(r'$U_e^2$')
    ax3.set_title(f"Population 02 - Excitatory Dynamics")

    ax4.set_xlabel(r'$K_1$'); ax1.set_ylabel(r'$I^1_{in}$'); ax4.set_zlabel(r'$U_i^2$')
    ax4.set_title(f"Population 02 - Inhibitory Dynamics")

    fig.suptitle(f"Fixed Point Evaluation K vs. " + r"$I^1_{in}$" + "\n" + r"($\sigma_e^1$ = " 
                 + f"{opts['sigma_e'][0]:.3f}mV, " + r"$\sigma_i^1$ = " + f"{opts['sigma_i'][0]:.3f}mV, " 
                 + r"$\sigma_e^2$ = " + f"{opts['sigma_e'][1]:.3f}mV, " + r"$\sigma_i^2$ = " + 
                 f"{opts['sigma_i'][1]:.3f}mV)\n", fontsize = 20)
    
    fig.tight_layout()

    file_str = f"sig_e1_{opts['sigma_e'][0]:.3f}_sig_i1_{opts['sigma_i'][0]:.3f}_sig_e2_{opts['sigma_e'][1]:.3f}_sig_i2_{opts['sigma_i'][1]:.3f}_3D.png"
    fig.savefig(os.path.join(save_path, file_str))