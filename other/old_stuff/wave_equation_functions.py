# -*- coding: utf-8 -*-
"""
Main Code of the Projektparktikum on Wave Equation

For Animations use >>%matplotlib qt<< in console or Tools>Preferences>IPython console>Graphics>Backend: Qt

"""
import numpy as np
from tqdm import tqdm  # animated progress bar for loops


# possible functions for initial values

def gaussian(x, xnull, a):
    return np.exp(-a * (x - xnull) ** 2)


def pulse(x, a, wavelength):
    return np.exp(-a * x ** 2) * np.sin(2 * np.pi / wavelength * x)


def sine(x, wavelength):
    return np.sin(2 * np.pi / wavelength * x)


def waveeq(phi0, pi0, x, periods, alpha, boundaries='p', outputstep=1):
    '''
    solver for wave equation with runge kutta 4 time evolution

    Arguments:
    ----------
        x           : 1d-array
            spatial grid
        phi0        : 1d-array
            initial value function of displacement
        pi0         : 1d-array
            initial value function of velocity
        periods     : int
            number of full periods for time evolution
        alpha       : float
            courant numberm, theory suggests 0<alpha<=1
        boundaries  : string
            'r' = rigid boundary conditions
            'ref'= reflecting boundary conditions
            'p' = periodic boundary conditions
            'o' = open boundary conditions
        outputstep  : int
            only write every 'outputstep' timestep to return array

    Returns:
    --------
        output      : 3d-array
            (phi(x),pi(x),t)
            axis 0 = x
            axis 1 = phi, pi
            axis 2 = t
            output[:,0,t] = phi(t)
            output[:,1,t] = pi(t)
        x           : 1d array
        t           : 1d array
    '''
    c = 1
    c2 = c ** 2

    Nx = len(x) - 1
    dx = (x[-1] - x[0]) / (Nx)
    dx2 = dx ** 2
    dt = alpha / c * dx
    Nt = int(periods) * int(c / alpha * (x[-1] - x[0]) * round(
        1 / dt))  # number of full evolutions through grid, when one evolution has same Nt as Nx
    tstart = 0
    tend = Nt
    t = np.linspace(tstart, tend, Nt)

    # initialize arrays for solution
    phi = np.zeros(shape=(Nx + 1, Nt + 1), dtype='float64')
    pi = np.zeros(shape=(Nx + 1, Nt + 1), dtype='float64')
    output = np.zeros(shape=(Nx + 1, Nx + 1, int((Nt) / outputstep + 1)),
                      dtype='float64')  # 3D array with (phi(x),pi(x),t)

    # set initial values for t=0 as first step in solution arrays

    phi[:, 0] = phi0

    # alternative 1: no momentum for symmetric propagation
    # pi[:,0]  = pi0
    # alternative 2: initial speed for foreward propagation
    pi[1:-1, 0] = -alpha / c * (phi0[2:] - phi0[:-2]) / (2 * dt)  # first order central stencil
    pi[0, 0] = -alpha / c * (phi0[1] - phi0[0]) / dt  # first order forward stencil
    pi[-1, 0] = -alpha / c * (phi0[-1] - phi0[-2]) / dt  # first order backward stencil

    output[:, 0, 0] = phi[:, 0]
    output[:, 1, 0] = pi[:, 0]

    for ti, timei in tqdm(enumerate(t)):
        # Simple Euler Method
        # phi[:,ti+1] = phi[:,ti]+pi[:,ti]*dt
        # pi[:,ti+1]  = pi[:,ti] + (Diff2(phi[:,ti],dx))*dt

        # Runge Kutta 4th order
        def F(phi_in, pi_in):
            F = np.ndarray(shape=(len(x), 2))
            if boundaries == 'o2':
                Ftemp = np.zeros((x.size, 2))  # no ghost points
                Ftemp[:, 0] = phi_in[:]
                Ftemp[:, 1] = pi_in[:]
            else:
                # initialize temporary array for output with added ghost points
                Ftemp = np.zeros((x.size + 2, 2))  # adding 2 ghost points
                Ftemp[1:-1, 0] = phi_in[:]  # fill center with original function; ghosts are ftemp[0] and ftemp[-1]
                Ftemp[1:-1, 1] = pi_in[:]

            # rigid boundaries
            if boundaries == 'r':
                '''
                aka Dirichlet boundary conditions
                phi(0) and phi(L) = 0
                is fulfilled by Ftemp = np.zeros(...), so Ftemp[0]=0 every function F call
                '''
                pass

            # reflecting boundaries
            if boundaries == 'ref':
                '''
                aka Neumann boundary conditions
                dphi/dx(0 & L) = 0
                (phi[L]-phi[L-1])/dx = 0 or phi[L] = phi[L-1]
                '''
                Ftemp[0, 0] = Ftemp[1, 0]
                Ftemp[-1, 0] = Ftemp[-2, 0]

            # periodic boundaries
            elif boundaries == 'p':
                '''
                phi[L] = phi[0]
                '''
                Ftemp[0, 0] = phi_in[-1]
                Ftemp[-1, 0] = phi_in[0]

            # open boundaries
            elif boundaries == 'o1':
                Ftemp[0, 0] = 2 * phi_in[0] - phi_in[1]
                Ftemp[-1, 0] = 2 * phi_in[-1] - phi_in[-2]

            # open boundaries
            elif boundaries == 'o2':
                Ftemp[0, 1] = phi_in[0] - alpha * (phi_in[1] - phi_in[0])
                Ftemp[-1, 1] = phi_in[-1] - alpha * (phi_in[-2] - phi_in[-1])

            else:
                raise ValueError('Invalid input: '
                                 'boundaries must either be "r", "p" or "o"')

            if boundaries == 'o2':
                F[:, 0] = Ftemp[:, 1]
                F[1:-1, 1] = c2 * (Ftemp[2:, 0] - 2 * Ftemp[1:-1, 0] + Ftemp[:-2, 0]) / dx2
                # F[0,1]=c2*(Ftemp[2,0]-2*Ftemp[1,0]+Ftemp[0,0])/dx2
                # F[-1,1]=c2*(Ftemp[-1,0]-2*Ftemp[-2,0]+Ftemp[-3,0])/dx2

            else:
                F[:, 0] = Ftemp[1:-1, 1]
                F[:, 1] = c2 * (Ftemp[2:, 0] - 2 * Ftemp[1:-1, 0] + Ftemp[:-2, 0]) / dx2
            return F

        K1 = F(phi[:, ti], pi[:, ti])[:, 0]
        L1 = F(phi[:, ti], pi[:, ti])[:, 1]

        K2 = F(phi[:, ti] + 0.5 * dt * K1, pi[:, ti] + 0.5 * dt * L1)[:, 0]
        L2 = F(phi[:, ti] + 0.5 * dt * K1, pi[:, ti] + 0.5 * dt * L1)[:, 1]

        K3 = F(phi[:, ti] + 0.5 * dt * K2, pi[:, ti] + 0.5 * dt * L2)[:, 0]
        L3 = F(phi[:, ti] + 0.5 * dt * K2, pi[:, ti] + 0.5 * dt * L2)[:, 1]

        K4 = F(phi[:, ti] + dt * K3, pi[:, ti] + dt * L3)[:, 0]
        L4 = F(phi[:, ti] + dt * K3, pi[:, ti] + dt * L3)[:, 1]

        phi[:, ti + 1] = phi[:, ti] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6
        pi[:, ti + 1] = pi[:, ti] + dt * (L1 + 2 * L2 + 2 * L3 + L4) / 6

        if ti == 0 or (ti) % outputstep == 0:
            output[:, 0, (ti) // outputstep] = phi[:, ti + 1]
            output[:, 1, (ti) // outputstep] = pi[:, ti + 1]
        else:
            pass
    return output, x, t
