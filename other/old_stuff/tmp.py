#!/usr/local/bin/python

""" Black-Hole perturbations
Time domain solution of the Regge-Wheeler-Zerilli equation
4th order accurate finite difference scheme
"""

__author__ = "S.Bernuzzi (Parma U & INFN)"
__copyright__ = "Copyright 2017"

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys


def tort2schw_root(rst, M=1, acc=1e-14, maxit=100):
    """
        Return r(r*) using a root finder
    :rst:  a point of the grid.
    """

    tm = 2. * M
    ootm = 1. / tm
    rmin = (2. + acc) * M
    # initial guess for the solution
    if rst < 0.:
        ro = rmin + 0.001
    elif rst < 4. * M:
        ro = 4.1 * M
    elif rst > 4. * M:
        ro = rst * M
    else:
        # rst == 4.*M
        return rst
    #


    for i in np.arange(maxit):
        # recursive algorithm
        # r* = r + 2 * G * M ln(-1. + r/2GM)
        rn = ro - ((tm - ro) * (rst - ro - tm * np.log(-1. + ro * ootm))) / ro
        rn = max(rn, rmin)
        delta = np.fabs(rn - ro)
        if np.fabs(delta) < acc:
            return rn
        ro = rn
    return -1


def tort2schw(rs, M=1., acc=1e-14, maxit=100):
    """
    :rs: array. Radial grid.
    Return r(r*) using a root finder
    """
    if M <= 0.:
        # if the space-time is flat, the schw radius = tort radius
        return rs
    r = np.zeros_like(rs)
    for i in np.arange(len(rs)):
        r[i] = tort2schw_root(rs[i], M, acc, maxit)
    return r


def retime(t, rs, M=1):
    """
    Retarded time
    """
    if M == 0.:
        return t
    return 0.5 * (t - rs) / M


def rwz_potential(r, l=2, M=1.):
    """
    Even/Odd RWZ potentials
    """
    if M == 0.:
        return np.zeros_like(r), np.zeros_like(r)
    lam = l * (l + 1.)
    lamm2 = lam - 2.
    lamm2sq = lamm2 ** 2
    r2 = r ** 2
    r3 = r2 * r
    oor = 1. / r
    oor2 = oor ** 2
    oor3 = oor ** 3
    A = (1. - 2. * M * oor)
    Ve = - A * (lam * lamm2sq * r3 + 6.0 * lamm2sq * M * r2 + 36. * lamm2 * M ** 2 * r + 72. * M ** 3) / (
                r3 * (lamm2 * r + 6. * M) ** 2)
    #
    Vo = - A * (lam * oor2 - 6. * M * oor3)
    return Ve, Vo


def rwz_rhs(u, (rs, V, nr, ng)):
    """
    r.h.s. of RWZ equation
    4th order finite differencing implementation
    """
    # unpack state vector
    N = nr + 2 * ng  # grid size including ghosts
    dotu = np.zeros_like(u)  # rhs
    # z = np.copy(u[0:N]) # copy (extra mem)
    # s = np.copy(u[N:2*N])
    z = u[0:N]  # reference
    s = u[N:2 * N]
    dotz = dotu[0:N]  # reference
    dots = dotu[N:2 * N]
    d2z = np.zeros_like(z)  # mem for drvt
    # shorthands
    drs = rs[1] - rs[0]
    factdrs2 = 1. / (12. * drs * drs)
    othree = 1. / 3.
    # fill ghosts for boundary conditions
    # maximally dissipative BCs
    # See Eq.(27)-(30) of Calabrese & Gundlach arxiv:0509119
    i = 1
    z[i] = - 4. * drs * s[i + 1] - 10. * othree * z[i + 1] + 6. * z[i + 2] - 2. * z[i + 3] + othree * z[i + 4]

    i = N - 2
    z[i] = - 4. * drs * s[i - 1] - 10. * othree * z[i - 1] + 6. * z[i - 2] - 2. * z[i - 3] + othree * z[i - 4]

    i = 0
    # z[i] = - 20.*drs*s[i+2] - 80.*othree*z[i+2] + 40.*z[i+3] - 15.*z[i+4] + 8.*othree*z[i+5]   --   32
    z[i] = + 5. * z[i + 1] - 10. * z[i + 2] + 10. * z[i + 3] - 5. * z[i + 4] + z[i + 5]
    # z[i] = - 20 * s[i + 2] - (80/3.) * z[i + 3] + 40 * z[i + 4] - 15. * z[i + 5] + (8/3.) * z[i+6]

    i = N - 1
    # z[i] = - 20.*drs*s[i-2] - 80.*othree*z[i-2] + 40.*z[i-3] - 15.*z[i-4] + 8.*othree*z[i-5]   --   32
    z[i] = + 5 * z[i - 1] - 10. * z[i - 2] + 10. * z[i - 3] - 5. * z[i - 4] + z[i - 5]
    # z[i] = 20 * s[i - 2] - (80 / 3.) * z[i - 3] + 40 * z[i - 4] - 15. * z[i -5] + (8 / 3.) * z[i - 6]

    # u = (z,s)
    # z,t = s
    # s,t = z,xx + V z
    i = np.arange(2, nr + 2)
    # Wiki: -1/12 * z[i-2] + 4/3 * z[i-1] -5/2 * z[i] + 4/3 * z[i+1] - 1/12 * z[i+2]
    d2z[i] = (16. * (z[i + 1] + z[i - 1]) - 30. * z[i] - (z[i + 2] + z[i - 2])) * factdrs2
    dotz[i] = s[i]
    dots[i] = d2z[i] + z[i] * V[i]
    # print(dotu); exit(1)
    return dotu


def timestep(n, dt, u, rhs, pars):
    """
    Runge-Kutta 4
    """
    # storage
    utmp = np.zeros_like(u)
    # shorthand
    halfdt = 0.5 * dt
    dt6 = dt / 6.
    # time = n*dt
    # steps
    k1 = rhs(u, pars)
    utmp[:] = u[:] + halfdt * k1[:]
    k2 = rhs(utmp, pars)
    utmp[:] = u[:] + halfdt * k2[:]
    k3 = rhs(utmp, pars)
    utmp[:] = u[:] + dt * k3[:]
    k4 = rhs(utmp, pars)
    u[:] = u[:] + dt6 * (k1[:] + 2. * (k2[:] + k3[:]) + k4[:])
    return u


def gaussian_packet(x, Rps, dsigma, timesym=0.):
    """
    Gaussian packet initial data
    """
    d2sigma = dsigma * dsigma
    norm = dsigma / np.sqrt(2.0 * np.pi)
    gauss = norm * np.exp(-0.5 * d2sigma * (x - Rps) ** 2)
    dgauss = -(x - Rps) * d2sigma * gauss * timesym
    return gauss, dgauss


def output_solution(fname, time, rs, z, di):
    """
    Output solution at given time,
    append if file exists
    """
    with open(fname, 'a') as f:
        # resample excluding ghosts, assume ng=2
        np.savetxt(f, np.c_[rs[2:-2:di], z[2:-2:di]], fmt='%.9e', header="time = %.6e" % time, comments='"')
        f.write(b'\n\n')
    return


def output_detector(fname, time, z):
    """
    Output wave at detector,
    append if file exists
    use index instead of radius for convergence studies)
    """
    with open(fname, 'a') as f:
        np.savetxt(f, np.c_[time, z], fmt='%.9e')
    return


def run(rsmin=-300,  # tortoise coordinate, grid boundaries
        rsmax=+300,  #
        nr=401,  # number of physics grid points
        cfl=0.5,  # CFL for timestep
        tend=600,  # final evol time
        ell=2,  # multipole
        M=1.,  # BH mass
        Rps=150.,  # Gaussian initial data
        dsigma=0.25,  # Gaussian sigma
        outdir="data/",  # where to save output, make but does not overwrite
        index_det=[2, 200 + 2, 400 + 2],  # index(es) for the detectors
        itout=(1, 10),  # output every 'itout' iters for 0d and 1d data
        isout=2,  # spatial output sampling
        ):
    """
    Main driver
    """

    if not os.path.isdir(outdir):
        os.mkdir(outdir, 0755)
    # grid & storage
    ng = 2          # ghosts (for 2nd order accuracy)
    N = nr + 2 * ng # all spatial points
    drs = (rsmax - rsmin) / float((nr - 1))     # grid spacing
    rs = -2 * drs + rsmin + drs * np.arange(N)  # total spatial
    dt = cfl * drs  # time steps
    u = np.zeros(2 * N)
    # print(rs); exit(1)
    #
    print "drs=%.4e" % drs
    print "dt =%.4e" % dt
    print "rdet =", rs[index_det]
    # exit(1)
    # Schw radius
    r = tort2schw(rs, M) # r grid

    ### PLOT COORDINATES
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(np.arange(len(r)), rs, color = 'black', label='tortous')
    # ax.plot(np.arange(len(r)), r, color='gray', label='swarchild')
    # plt.legend()
    # plt.show()
    # plt.close()

    # exit(1)
    # Potential
    Ve, Vo = rwz_potential(r, ell, M)
    if ell % 2 == 0:
        V = Ve
    else:
        V = Vo
    np.savetxt(outdir + "rwz_potential.txt", np.c_[rs, r, -Ve, -Vo], fmt='%.8e')
    #
    beta = 1j * 1 * np.pi / 2
    kappa = 0.2
    v0 = 1.
    V = v0 / ((np.exp(kappa * rs + beta) - np.exp(-kappa * rs - beta)) / (
                np.exp(kappa * rs + beta) + np.exp(-kappa * rs - beta))) ** 2 - v0

    # initial data
    (z, s) = gaussian_packet(rs, Rps, dsigma)
    np.savetxt(outdir + "rwz_initialdata.txt", np.c_[rs, z, s], fmt='%.8e')
    u[0:N] = z[:]
    u[N:2 * N] = s[:]
    # time evolution
    itmin = min(itout)
    t = 0
    i = 0
    _i = 0
    while t < tend:
        print("t:{:07d} t:{:.4e} / {:.4e}".format(i, t, tend))

        if i % itout[0] == 0:
            z[:] = u[0:N]
            s[:] = u[N:2 * N]
            for idet in index_det:
                output_detector(outdir + "psi_%06d" % idet + ".txt", retime(t, rs[idet]), z[idet])
                output_detector(outdir + "dpsi_%06d" % idet + ".txt", retime(t, rs[idet]), s[idet])
                print(_i)
                _i = _i + 1
        if i % itout[1] == 0:
            z[:] = u[0:N]
            s[:] = u[N:2 * N]
            output_solution(outdir + "rwz.yg", t, rs, z, isout)
        i = i + 1
        t = i * dt
        u = timestep(i, dt, u, rwz_rhs, (rs, V, nr, ng))
    return


if __name__ == "__main__":

    if 1:
        #
        # run a self-convergence test in flat spacetime
        #
        mass = 0  # flat spacetime
        run(-1, +1, 101, 0.5, 1.2, 2, mass, 0., 10., "flat_101/", [2 + 50], [2, 20], 1)
        # run(-1, +1, 201, 0.5, 1.2, 2, mass, 0., 10., "flat_201/", [2 + 100], [4, 40], 2)
        # run(-1, +1, 401, 0.5, 1.2, 2, mass, 0., 10., "flat_401/", [2 + 200], [8, 80], 4)
        # pick solution at detectors x=0.
        t, low = np.loadtxt("flat_101/psi_000052.txt", unpack=True)
        t, med = np.loadtxt("flat_201/psi_000102.txt", unpack=True)
        t, hig = np.loadtxt("flat_401/psi_000202.txt", unpack=True)
        print("low:{} med:{} high:{}".format(len(low), len(med), len(hig)))
        plt.plot(t, low - med, 'o-', label="(low-med)")
        plt.plot(t, med - hig, 'x-', label="(med-hig)")
        plt.plot(t, (low - med) / 16., '--', label="(low-med) scaled 4th order")
        plt.xlabel('time')
        plt.legend()
        plt.show()

    #
    # BH run
    #
    run()

    rs, r, Ve, Vo = np.loadtxt("data/rwz_potential.txt", unpack=True)
    plt.plot(rs, Ve, '-')
    plt.xlabel(r'$r_*$')
    plt.ylabel(r'$V(\ell=2)$')
    plt.xlim([-20, 50])
    plt.savefig("data/potential.png")

    t, psi = np.loadtxt("data/psi_000402.txt", unpack=True)
    fig, ax1 = plt.subplots()
    ax1.plot(t, psi, '-')
    ax1.set_xlabel(r'$u$')
    ax1.set_ylabel(r'$\Psi_{20}$')
    ax2 = ax1.twinx()
    ax2.semilogy(t, np.fabs(psi), 'r-')
    ax2.set_ylabel(r'|$\Psi_{20}|$')
    # ax1.set_ylim([-0.04,0.05])
    ax2.set_ylim([1e-7, 1e-1])
    plt.xlim([0, 150])
    fig.tight_layout()
    plt.savefig("data/psi.png")
    plt.show()


