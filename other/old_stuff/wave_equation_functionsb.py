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
            'o' = open boundary conditions, 1..3
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

    # c is not thoroughly implemented
    c = 1
    c2 = c ** 2

    Nx = len(x) - 1
    dx = (x[-1] - x[0]) / (Nx)
    dx2 = dx ** 2
    dt = alpha / c * dx
    Nt = int(periods) * int((x[-1] - x[0]) * round(
        1 / dt)) + 2  # number of full evolutions through grid, when one evolution has same Nt as Nx
    tstart = 0
    tend = Nt
    t = np.linspace(tstart, tend, Nt)

    # initialize arrays for solution
    # only 2 timesteps to save memory
    phi = np.zeros(shape=(Nx + 1, 2), dtype='float64')
    pi = np.zeros(shape=(Nx + 1, 2), dtype='float64')
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
        def F(phi_in, pi_in):
            '''
            wave equation function

            d/dt (phi,pi) = F (phi,pi)

            Parameters
            ----------
            phi_in : 1d-array
                displacement function.
            pi_in : 1d-array
                velocity function.

            Raises
            ------
            ValueError
                for misspelled boundary condition input.

            Returns
            -------
            F : 2d-array
                F [:,0] = phi_out
                F [:,1] = pi_out.

            '''
            F = np.ndarray(shape=(len(x), 2))

            ###############
            # GHOSTPOINTS #
            ###############
            # initialize temporary array for output with added ghost points
            Ftemp = np.zeros((x.size + 2, 2))  # adding 2 ghost points
            Ftemp[1:-1, 0] = phi_in[:]  # fill center with original function; ghosts are ftemp[0] and ftemp[-1]
            Ftemp[1:-1, 1] = pi_in[:]

            #######################
            # BOUNDARY CONDITIONS #
            #######################

            # rigid boundaries
            if boundaries == 'r':
                '''
                aka Dirichlet boundary conditions
                phi(0) and phi(L) = 0
                is fulfilled by Ftemp = np.zeros(...), so Ftemp[0]=Ftemp[-1] = 0 every function F call
                '''
                pass

            # reflecting boundaries
            elif boundaries == 'ref':
                '''
                aka Neumann boundary conditions
                dphi/dx(0 & L) = 0 discretized with forward/backward stencil
                (phi[L]-phi[L-1])/dx = 0 or phi[L] = phi[L-1]

                alternatively with central stencil
                (phi[L]-phi[L-2])/(2dxO) = 0 or phi[L] = phi[L-2]
                '''
                Ftemp[0, 0] = Ftemp[1, 0]
                Ftemp[-1, 0] = Ftemp[-2, 0]

                # alternative
                # Ftemp[0,0]    = Ftemp[2,0]
                # Ftemp[-1,0]   = Ftemp[-3,0]

            # periodic boundaries
            elif boundaries == 'p':
                '''
                phi[L] = phi[0]
                '''
                Ftemp[0, 0] = phi_in[-1]
                Ftemp[-1, 0] = phi_in[0]
                Ftemp[0, 1] = pi_in[-1]
                Ftemp[-1, 1] = pi_in[0]

            # open boundaries (1) - simple extrapolation
            elif boundaries == 'o1':
                Ftemp[0, 0] = 2 * phi_in[0] - phi_in[1]
                Ftemp[-1, 0] = 2 * phi_in[-1] - phi_in[-2]
                Ftemp[0, 1] = 2 * pi_in[0] - pi_in[1]
                Ftemp[-1, 1] = 2 * pi_in[-1] - pi_in[-2]

            # open boundaries (2) - discretization dx phi +- dt phi = 0
            elif boundaries == 'o2':
                Ftemp[0, 0] = phi_in[0] - alpha * (phi_in[1] - phi_in[0])
                Ftemp[-1, 0] = phi_in[-1] - alpha * (phi_in[-2] - phi_in[-1])
                pass

            # open boundaries (3) - cancelling characteristics in phi and pi
            elif boundaries == 'o3':
                Ftemp[0, 0] = -Ftemp[1, 1] * 2 * dx / c + Ftemp[2, 0]
                Ftemp[-1, 0] = -Ftemp[-2, 1] * 2 * dx / c + Ftemp[-3, 0]
            else:
                raise ValueError('Invalid input: '
                                 'boundaries must either be "r", "ref", "p", "o1", "o2" or "o3"')

            ############
            # OUTPUT F #
            ############
            F[:, 0] = Ftemp[1:-1, 1]
            F[:, 1] = c2 * (Ftemp[2:, 0] - 2 * Ftemp[1:-1, 0] + Ftemp[:-2, 0]) / dx2
            return F

        # Simple Euler Method
        # phi[:,ti+1] = phi[:,ti]+pi[:,ti]*dt
        # pi[:,ti+1]  = pi[:,ti] + F(phi[:,ti],pi[:,ti])[:,1]*dt

        # #Runge Kutta 4th order
        K1 = F(phi[:, 0], pi[:, 0])[:, 0]
        L1 = F(phi[:, 0], pi[:, 0])[:, 1]

        K2 = F(phi[:, 0] + 0.5 * dt * K1, pi[:, 0] + 0.5 * dt * L1)[:, 0]
        L2 = F(phi[:, 0] + 0.5 * dt * K1, pi[:, 0] + 0.5 * dt * L1)[:, 1]

        K3 = F(phi[:, 0] + 0.5 * dt * K2, pi[:, 0] + 0.5 * dt * L2)[:, 0]
        L3 = F(phi[:, 0] + 0.5 * dt * K2, pi[:, 0] + 0.5 * dt * L2)[:, 1]

        K4 = F(phi[:, 0] + dt * K3, pi[:, 0] + dt * L3)[:, 0]
        L4 = F(phi[:, 0] + dt * K3, pi[:, 0] + dt * L3)[:, 1]

        phi[:, 1] = phi[:, 0] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6
        pi[:, 1] = pi[:, 0] + dt * (L1 + 2 * L2 + 2 * L3 + L4) / 6

        if ti == 0 or (ti) % outputstep == 0:
            output[:, 0, (ti) // outputstep] = phi[:, 1]
            output[:, 1, (ti) // outputstep] = pi[:, 1]
        else:
            pass

        # prepare for next iteration
        phi[:, 0] = phi[:, 1]
        pi[:, 0] = pi[:, 1]

    return output, x, t
