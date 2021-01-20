#
import numpy as np
import matplotlib.pyplot as plt


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



x = np.mgrid[-300:300:100j]

rhw_potential, _= rwz_potential(tort2schw(x), l=2, M=1.)


beta = 1j * 1 * np.pi / 2
kappa = 0.2
v0 = 1.
V = v0 / ((np.exp(kappa * x + beta) - np.exp(-kappa * x  - beta)) / (np.exp(kappa * x + beta) + np.exp(-kappa * x  - beta))) ** 2 - v0

a = -0.
b = -1.
kappa = 1/2.
V = (a / ((np.exp(kappa * x) +  np.exp(-kappa * x))/2.)) + (b / ((np.exp(kappa * x) - np.exp(-kappa * x)) / 2.))

# k = 0.000001
# V = - k*x ** 2


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, rhw_potential, '-', color='black')
ax.plot(x, V, '-', color='gray')
plt.show()