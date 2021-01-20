# -*- coding: utf-8 -*-
"""
Convergence for the Projektparktikum on Wave Equation

For Animations use >>%matplotlib qt<< in console or Tools>Preferences>IPython console>Graphics>Backend: Qt

"""
import numpy as np
import matplotlib.pyplot as plt
from wave_equation_functionsb import *
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# set input

alpha = 0.5  # Currant Number should be less than 1
boundaries = 'p'

# initialize grid
# Nx_lst = [200, 400, 800, 1600, 3200]
Nx_lst = [200, 400, 800, 1600, 3200, 6400, 12800]
acc = np.ndarray((min(Nx_lst) + 1, len(Nx_lst)))
error_acc = np.zeros((min(Nx_lst), len(Nx_lst) - 1))
error_self = np.zeros((min(Nx_lst), len(Nx_lst) - 1))

for i, Nxi in enumerate(Nx_lst):
    Nx = Nxi
    xstart = -5
    xend = 5
    dx = (xend - xstart) / (Nx)
    # avoid zeros Avoid common multiples with transcendent factor 1/pi
    # x=np.linspace(xstart+dx/np.pi,xend+dx/np.pi,Nx+1)
    x = np.linspace(xstart, xend, Nx + 1)

    periods = 2

    step = int(2 * (x[-1] - x[0]) * round(1 / dx) + 2)

    # create array with initial values, i.e. choose a function
    phi0 = gaussian(x, 0, 2)
    # phi0 = sine(x,2)
    # phi0  = pulse(x,1,10)
    pi0 = np.zeros_like(x)

    # call function
    output, x, t = waveeq(phi0, pi0, x, periods, alpha, boundaries, step)

    phi = output[:, 0, :]
    pi = output[:, 1, :]

    acc[:, i] = phi[::2 ** i, 1]

    fig, ax = plt.subplots()
    for n in range(len(output[0, 0, :])):
        label = "period %i" % (n)
        ax.plot(x, phi[:, n], label=label)
        ax.vlines(x=0, ymin=0, ymax=1)
        ax.legend()
    plt.show()

    # error = np.zeros((len(x),len(phi[0,:])))
    # for i in range(len(phi[0,:])):
    #     error[:,i] = (phi[:,i]-phi[:,0])/phi[:,0]

    # fig, ax = plt.subplots()
    # for n in range(len(output[0,0,:])):
    #     label = "period %i" % (n)
    #     ax.plot(x[100:-100], error[100:-100,n], label=label)
    #     ax.vlines(x=0, ymin=-0.1, ymax=0.1)
    #     ax.legend()
    # plt.show()

error_acc = np.zeros((min(Nx_lst) + 1, len(acc[0, :])))
for i, Nxi in enumerate(Nx_lst):
    error_acc[:, i] = (acc[:, i] - acc[:, -1]) / acc[:, -1]
fig, ax = plt.subplots()

for i, Nxi in enumerate(Nx_lst[:-1]):
    label = "grid = %i" % (Nxi)
    ax.plot(error_acc[:, i], label=label)
    #    ax.vlines(x=0, ymin=-0.1, ymax=0.1)
    ax.legend()
plt.show()

selfconv = np.zeros((2, len(acc[0, :]) - 2))
for i, Nxi in enumerate(Nx_lst[:-2]):
    selfconv[1, i] = np.sqrt(np.sum(np.abs(acc[:, i] - acc[:, i + 1]) ** 2)) / \
                     np.sqrt(np.sum(np.abs(acc[:, i + 1] - acc[:, i + 2]) ** 2))
    selfconv[0, i] = (xend - xstart) / Nxi

fig, ax = plt.subplots()
# for i, Nxi in enumerate(Nx_lst[:-2]):
# label = "grid = %i" % (selfconv[0,:])
ax.plot(selfconv[0, :], selfconv[1, :], '.--', label=label)
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_title(r'self convergence')
ax.set_xlabel(r'step size $h$ [a.u.]')
ax.set_ylabel(r'$\displaystyle \frac{|\phi^{h}-\phi^{h/2}|}{|\phi^{h/2}-\phi^{h/4}|}$ [a.u.]')
# ax.set_yscale("log")
# ax.legend()
plt.show()
