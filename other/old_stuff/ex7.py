#
# Algorithm implementing the wave equation
# with external potential
#
#

import numpy as np
import os
import matplotlib.pyplot as plt

''' potentials '''

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
def rwz_potential(xarr, ell = 2, l=2, M=1.):
    """
    Potential is defined for Swarchild coordinates.
    It exist as an 'odd' and 'eve' potential, Vo, Ve
    First the coorinate transformation is required
    :ell: ellepticity
    :param xarr: tortous coordinates
    :param l: multipole
    :param M: mass of a BH
    :return: v
    """

    r = tort2schw(xarr, M=M) # swarchild coordinates

    if M == 0.:
        # flat space-time
        return np.zeros(len(xarr))#, np.zeros_like(xarr)

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

    if ell % 2 == 0:
        V = Ve
    else:
        V = Vo

    return V

def pw_potential(xarr, m=1.):

    beta = 0#1j * 1 * np.pi / 2
    kappa = 0.1#0.1
    v0 = 0.15#0.15
    V = v0 * ((np.exp(kappa * xarr + beta) - np.exp(-kappa * xarr - beta)) /
              (np.exp(kappa * xarr + beta) + np.exp(-kappa * xarr - beta))) ** 2 - v0

    return V

''' initial data '''

def gaussian_packet(x, Rps, dsigma, timesym=0.):
    """
    Gaussian packet initial data
    """
    d2sigma = dsigma * dsigma
    norm = dsigma / np.sqrt(2.0 * np.pi)
    gauss = norm * np.exp(-0.5 * d2sigma * (x - Rps) ** 2)
    dgauss = -(x - Rps) * d2sigma * gauss * timesym
    return gauss, dgauss

''' other '''

def retime(t, rs, M=1.):
    """
    Retarded time
    """
    if M == 0.:
        # for flat space-time
        return t
    return 0.5 * (t - rs) / M

def output_detector(fname, time, z):
    """
    Output wave at detector,
    append if file exists
    use index instead of radius for convergence studies)
    """
    with open(fname, 'a') as f:
        np.savetxt(f, np.c_[time, z], fmt='%.9e')
    return

def output_solution(fname, time, rs, z, di, ng=2):
    """
    Output solution at given time,
    append if file exists
    """
    with open(fname, 'a') as f:
        # resample excluding ghosts, assume ng
        np.savetxt(f, np.c_[rs[ng:-ng:di], z[ng:-ng:di]], fmt='%.9e', header="time = %.6e" % time, comments='"')
        # f.write(b'\n\n')
    return

''' solver '''

class WaveEqSolver:

    def __init__(self, xmin=-300., xmax=300., nr=401, cfl=0.5, tend=600., mass=0.,
                 indexdet=(0,200,400), itout=(1,10), isout=2,
                 stencilorder=4, potential="RWZ", initdata="Gauss", outdir="./wave/"):
        #
        # xmin = -300. # coord min
        # xmax = 300.  # coord max
        # nr = 401     # phyiscal points
        # cfl = 0.5    # dt/dr
        # tend = 600.  # end of the evolution time
        # outdir = "./ex6/"
        # indexdet = [0,200,400]
        # itout = 1, 10     # save every nth iteration (for_detector, for xy graph)
        # isout = 2    # save every nth grid point
        # stencilorder=4
        # potential = "PT" # None or RWZ or PW
        # # M = 1.      # mass of the black hole
        # initdata = "Gauss"

        #
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        #
        if stencilorder == 2:
            rhs = self.rhs_stencil2
            ng = 1
        elif stencilorder == 4:
            rhs = self.rhs_stencil4
            ng = 2
        else:
            raise NameError("stencilorder={} is not recognized".format(stencilorder))
        indexdet = np.array(indexdet, dtype=int) + ng # shifting the detector with respect to ng
        #
        alln = nr + 2 * ng # all points of a grid
        dxs = (xmax - xmin) / float(nr-1) # grid spacing
        xs = -2. * dxs + xmin + dxs * np.arange(alln) # all spatial grid (+ghosts)
        dt = cfl * dxs # time step
        u = np.zeros(2 * alln) # solution vector. u[:N] = solution, u[N:2N] = its derivative
        #

        if potential == None:
            v = np.zeros(alln) # no potential. Flat space-time
        elif potential == "RWZ":
            v = rwz_potential(xs, ell=2, l=2, M=mass) # defauled RWZ potential
        elif potential == "PT":
            v = pw_potential(xs, mass) # defauled pochi-teller potential
        else:
            raise NameError("potential:{} is not implemented".format(potential))

        #
        if initdata == "Gauss":
            # filling the state vector with initial data
            u[0:alln], u[alln:2*alln] = gaussian_packet(xs, Rps=150., dsigma=0.25, timesym=0.)
        else:
            raise NameError("initdata:{} is not recognized".format(initdata))
        #

        t = 0
        i = 0
        _i = 0

        while t < tend:
            print("t:{:07d} t:{:.4e} / {:.4e}".format(i, t, tend))
            if i % itout[0] == 0:
                for idet in np.array([indexdet], dtype=int).flatten():
                    z = u[0:alln]
                    s = u[alln:2 * alln]
                    output_detector(outdir + "psi_%06d" % idet + ".txt",
                                    retime(t, xs[idet], mass),
                                    z[idet])
                    output_detector(outdir + "dpsi_%06d" % idet + ".txt",
                                    retime(t, xs[idet], mass),
                                    s[idet])
                    # print(_i)
                    _i = _i + 1

            if i % itout[1] == 0:
                z = u[0:alln]
                output_solution(outdir + "rwz.yg", t, xs, z, isout, ng = ng)
            i = i + 1
            t = i * dt
            u = self.timestep(i, dt, u, rhs, (xs, v, nr, ng))


    def timestep(self, i, dt, u, rhs, pars):
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
        k1 = rhs(u, *pars)
        utmp[:] = u[:] + halfdt * k1[:]
        k2 = rhs(utmp, *pars)
        utmp[:] = u[:] + halfdt * k2[:]
        k3 = rhs(utmp, *pars)
        utmp[:] = u[:] + dt * k3[:]
        k4 = rhs(utmp, *pars)
        u[:] = u[:] + dt6 * (k1[:] + 2. * (k2[:] + k3[:]) + k4[:])
        return u

    def rhs_stencil2(self, u, xs, v, nr, ng):
        N = nr + 2 * ng  # grid size including ghosts
        dotu = np.zeros_like(u)  # rhs

        z = u[0:N]  # reference
        s = u[N:2 * N]
        dotz = dotu[0:N]  # reference
        dots = dotu[N:2 * N]
        d2z = np.zeros_like(z)  # mem for drvt

        drs = xs[1] - xs[0]
        idx = 1. / drs

        i = 0
        z[i] = z[i + 2] - 2 * drs * s[i + 1]
        s[i] = 2 * s[i + 1] - s[i + 2]

        i = N - 1
        z[i] = z[i - 2] - 2 * drs * s[i - 1]
        s[i] = 2 * s[i - 1] - s[i - 2]

        i = np.arange(1, nr + 1)
        # Wiki: -1/12 * z[i-2] + 4/3 * z[i-1] -5/2 * z[i] + 4/3 * z[i+1] - 1/12 * z[i+2]
        d2z[i] = idx * idx * (z[i+1] + z[i-1] - 2 * z[i])
        dotz[i] = s[i]
        dots[i] = d2z[i] + z[i] * v[i]
        # print(dotu); exit(1)
        return dotu

    def rhs_stencil4(self, u, xs, v, nr, ng):
        """
        :param u: state vector u[0:N] - function, u[N:2N] - its derivative
        :return:
        """
        N = nr + 2 * ng  # grid size including ghosts
        dotu = np.zeros_like(u)  # rhs

        z = u[0:N]  # reference
        s = u[N:2 * N]
        dotz = dotu[0:N]  # reference
        dots = dotu[N:2 * N]
        d2z = np.zeros_like(z)  # mem for drvt

        drs = xs[1] - xs[0]
        factdrs2 = 1. / (12. * drs * drs)
        othree = 1. / 3.

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
        dots[i] = d2z[i] + z[i] * v[i]

        # print(dotu); exit(1)
        return dotu

def compare_potentials(xmin=-50., xmax=50., nr=401, mass=1.):

    # xs = np.arange(start=xmin, stop=xmax, step=nr)
    xs = np.mgrid[xmin:xmax:nr*1j]

    v_rwz = rwz_potential(xs, ell=2, l=2, M=mass)
    v_pt = pw_potential(xs, mass)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, v_rwz, color='red', ls='-', label="PWZ")
    ax.plot(xs, v_pt, color='blue', ls='-', label='PT')
    ax.set_xlabel(r'$xs$', fontsize='large')
    ax.set_ylabel(r'$V$', fontsize='large')
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.minorticks_on()
    ax.legend()
    fig.tight_layout()
    plt.savefig("./potential_comparison.png")
    plt.show()

def self_convergence():

    mass = 0
    WaveEqSolver(-300., 300, 101, 0.5, 600., mass, (50), (2, 20), 1,
                 stencilorder=4, potential="RWZ", initdata="Gauss", outdir="./flat_101/")
    WaveEqSolver(-300., 300, 201, 0.5, 600., mass, (100), (4, 40), 2,
                 stencilorder=4, potential="RWZ", initdata="Gauss", outdir="./flat_201/")
    WaveEqSolver(-300., 300, 401, 0.5, 600., mass, (200), (8, 80), 4,
                 stencilorder=4, potential="RWZ", initdata="Gauss", outdir="./flat_401/")

    # pick solution at detectors x=0.
    tl, low = np.loadtxt("flat_101/psi_000052.txt", unpack=True)
    tm, med = np.loadtxt("flat_201/psi_000102.txt", unpack=True)
    th, hig = np.loadtxt("flat_401/psi_000202.txt", unpack=True)
    print("low:{} med:{} high:{}".format(len(low), len(med), len(hig)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(tl, low - med, 'o-', label="(low-med)")
    ax.plot(tm, med - hig, 'x-', label="(med-hig)")
    ax.plot(th, (low - med) / 16., '--', label="(low-med) scaled 4th order")
    ax.set_xlabel('time')
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    # ax.set_yscale("log")
    ax.minorticks_on()
    ax.legend()
    plt.show()
    exit(1)

def find_nearest_index(array, value):
    ''' Finds index of the value in the array that is the closest to the provided one '''
    idx = (np.abs(array - value)).argmin()
    return idx

def load_solution(fname):

    if not os.path.isfile(fname):
        raise IOError("file:{} not found".format(fname))

    times = []
    # data = np.zeros(2)
    data = []
    with open(fname) as infile:
        for line in infile:
            if line.__contains__("\"time = "):
                t = float(line.split("\"time = ")[-1])
                times.append(t)
                print("read t: {}".format(t))
            elif len(line.split()) == 0:
                pass
            else:
                _row = np.array(line.split(), dtype=float)
                # print(_row)
                data.append(_row)#np.vstack((data, _row))
    # data = np.delete(data, 0, 0)
    data = np.array(data)
    times = np.array(times, dtype=float)

    # print(data.shape)
    # print(len(times))
    # print(len(data[:,0]) / len(times))
    data = np.reshape(data, ( len(times), int(len(data[:,0])/(len(times))), 2 ))
    print("Read times: {}".format(len(times)))
    print("Data:       {} [timesteps, data_coordiantes, data_values]".format(data.shape))
    # print(data[1, :, :])

    #

    #

    return times, data

def plot_solution_at_times():

    # list_times = [75, 100., 125., 150.]
    list_times = [150, 175., 200., 225.]
    # list_times = [75, 100., 125., 150.]
    list_fnames = ["./wave_rwz/rwz.yg"]
    colors = ["blue"]
    lss = ["-"]
    #
    res = {}
    #
    print("Collecting data")
    for fname in list_fnames:
        res[fname] = {}
        times, data = load_solution(fname)
        for t in list_times:
            res[fname][t] = {}
            idx = find_nearest_index(times, t)
            x_arr = data[idx, :, 0]
            y_arr = data[idx, :, 1]
            res[fname][t]["x_arr"] = x_arr
            res[fname][t]["y_arr"] = y_arr
    print("Data collected")
    #
    fig = plt.figure(figsize=(len(list_times)*3.6, 3.6))
    #
    axes = []
    for i in range(1, len(list_times)+1):
        if i == 1:axes.append(fig.add_subplot(1,len(list_times), i))
        else: axes.append(fig.add_subplot(1,len(list_times), i, sharey=axes[0]))
    #
    for ax, t in zip(axes, list_times):
        for fname, color in zip(list_fnames, colors):
            x_arr = res[fname][t]["x_arr"]
            y_arr = res[fname][t]["y_arr"]
            ax.plot(x_arr, y_arr, color=color, ls='-', label='{}'.format(fname))
    #
    for ax, time in zip(axes, list_times):
        ax.set_title("$time:{}$".format(time))
        ax.set_xlabel("$x$", fontsize='large')
        ax.set_ylabel("$\phi$", fontsize='large')
        ax.tick_params(
            axis='both', which='both', labelleft=True,
            labelright=False, tick1On=True, tick2On=True,
            labelsize=int(12),
            direction='in',
            bottom=True, top=True, left=True, right=True
        )
        ax.set_xlim(0, x_arr.max())
        ax.minorticks_on()
        ax.legend(loc="lower left", ncol=1)
        # ax.set_ylim(-2.,2.)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.0)
    plt.savefig("./profiles.png", dpi=128)
    plt.show()

def plot_solution_3d():

    list_fnames = ["./wave_rwz/rwz.yg", "./wave_pt/rwz.yg"]
    colors = ["blue", "red"]
    lss = ["-"]
    plot_every = 10 # plot every timestep

    res = {}
    _times = []
    print("Collecting data")
    for fname in list_fnames:
        res[fname] = {}
        times, data = load_solution(fname)
        # print(times[:]); exit()
        for t in times[::plot_every]:
            res[fname][t] = {}
            print("t:{}".format(t))
            idx = find_nearest_index(times, t)
            x_arr = data[idx, :, 0]
            y_arr = data[idx, :, 1]
            res[fname][t]["x_arr"] = x_arr
            res[fname][t]["y_arr"] = y_arr
    print("Data collected")
    # print(res[list_fnames[0]].keys())


    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for fname, color in zip(list_fnames, colors):
        for i, time in enumerate(res[list_fnames[0]].keys()):
            #
            x_arr = res[fname][time]["x_arr"]
            y_arr = res[fname][time]["y_arr"]

            z = np.zeros(x_arr.shape)
            z.fill(time)
            print("plotting t:{}".format(time))
            ax.plot(x_arr, y_arr, z, color=color, ls='-', lw=0.8)



    ax.legend(loc='lower left', shadow=False, fontsize='large', frameon=False)
    # ax.set_yscale("log")
    ax.set_xlabel("$x$", fontsize='large')
    ax.set_ylabel("$\phi$", fontsize='large')
    ax.set_zlabel("$time$", fontsize='large')
    # ax.set_ylabel()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.minorticks_on()
    ax.set_title("Wave propagation")
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig("./wave_3d.png", dpi=128)
    plt.tight_layout()
    plt.show()
    exit(1)

if __name__ == '__main__':

    arr = np.array([1,2,3,4,5,6,7,8,9], dtype=int)
    print(arr)
    arr = arr[3:]
    print(arr)
    # exit(1)

    # compare_potentials()
    #
    # self_convergence()
    #
    # plot_solution_at_times()
    #
    plot_solution_3d()
    # exit(1)
    # load_solution("./wave_rwz/rwz.yg")
    #
    # WaveEqSolver(-300., 300., 401, 0.5, 600., 1., (0, 200, 400), (1, 10), 2,
    #              stencilorder=4, potential="RWZ", initdata="Gauss", outdir="./wave_rwz/")
    #
    WaveEqSolver(-300., 300., 401, 0.5, 600., 1., (0, 200, 400), (1, 10), 2,
                 stencilorder=4, potential="PT", initdata="Gauss", outdir="./wave_pt/")

    t1, psi_rwz = np.loadtxt("./wave_rwz/psi_000402.txt", unpack=True)
    t2, psi_pt = np.loadtxt("./wave_pt/psi_000402.txt", unpack=True)
    fig, ax1 = plt.subplots()
    ax1.plot(t1, psi_rwz, '-', color='red', label='RWZ')
    ax1.plot(t2, psi_pt, '-', color='blue', label='PT')
    ax1.set_xlabel(r'$u$',  fontsize='large')
    ax1.set_ylabel(r'$\Psi_{20}$',  fontsize='large')
    ax1.legend()
    ax1.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=False,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax1.minorticks_on()

    ax2 = ax1.twinx()
    ax2.semilogy(t1, np.fabs(psi_rwz), '--', color='red')
    ax2.semilogy(t2, np.fabs(psi_pt), '--', color='blue')
    ax2.set_ylabel(r'|$\Psi_{20}|$',  fontsize='large')
    # ax1.set_ylim([-0.04,0.05])
    ax2.set_ylim([1e-7, 1e-1])
    ax2.tick_params(
        axis='both', which='both', labelleft=False,
        labelright=True, tick1On=False, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax2.minorticks_on()

    plt.xlim([0, 150])
    fig.tight_layout()
    plt.savefig("./psi.png")
    plt.show()