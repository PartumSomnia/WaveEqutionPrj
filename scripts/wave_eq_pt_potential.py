import numpy as np
import matplotlib.pyplot as plt

FIGPATH = "../figs/wave_eq_pt_potential/"

class RHS:
    """
        Different RHS for wave equation for
        2nd and 4th order accuracy finite differencing
    """

    @staticmethod
    def rhs_stencil2(u, xs, v, nr, ng):
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

    @staticmethod
    def rhs_stencil4(u, xs, v, nr, ng):
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

def RK4(dt, u, rhs, pars):
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

def PT_potential(xarr):
    """
    standart formula for PT potential, where hyperbolic sin
    is expressed as exponentials
    :param xarr:
    :return:
    """

    beta = 0#1j * 1 * np.pi / 2
    kappa = 0.1#0.1
    v0 = 0.15#0.15
    V = v0 * ((np.exp(kappa * xarr + beta) - np.exp(-kappa * xarr - beta)) /
              (np.exp(kappa * xarr + beta) + np.exp(-kappa * xarr - beta))) ** 2 - v0

    return V

def gaussian_packet(x, Rps, dsigma, timesym=0.):
    """
    Gaussian packet initial data
    """
    d2sigma = dsigma * dsigma
    norm = dsigma / np.sqrt(2.0 * np.pi)
    gauss = norm * np.exp(-0.5 * d2sigma * (x - Rps) ** 2)
    dgauss = -(x - Rps) * d2sigma * gauss * timesym
    return gauss, dgauss

def solver(xmin=-300., xmax=300.,
           nx=401, cfl=0.5, tend=600., itout=(1,10), isout=2,
           indexdet=400, stencilorder=2,
           potential=True, return_full=False):
    """
        Solves the wave equation evolution for given parameters
        outputs either solution at a detector or full solution on the grid
    """

    if stencilorder == 2:
        rhs = RHS.rhs_stencil2
        ng = 1
    elif stencilorder == 4:
        rhs = RHS.rhs_stencil4
        ng = 2

    # shifting the detector with respect to ng
    indexdet = indexdet + ng

    # grid
    alln = nx + 2 * ng  # all points of a grid
    dxs = (xmax - xmin) / float(nx - 1)  # grid spacing
    xs = -2. * dxs + xmin + dxs * np.arange(alln)  # all spatial grid (+ghosts)
    dt = cfl * dxs  # time step
    u = np.zeros(2 * alln)  # solution vector. u[:N] = solution, u[N:2N] = its derivative

    # potential
    if potential: v = PT_potential(xs)
    else: v = np.zeros_like(xs)

    # initial data
    u[0:alln], u[alln:2 * alln] = gaussian_packet(xs, Rps=150., dsigma=0.25, timesym=0.)

    # main loop
    t = 0
    i = 0

    solution_det = []
    solution = []
    solution_t = []
    while t < tend:
        print("t:{:07d} t:{:.4e} / {:.4e}".format(i, t, tend))
        if i % itout[0] == 0:
            # output solution at a detector
            z = u[0:alln]
            s = u[alln:2 * alln]
            solution_det.append([t, z[indexdet], s[indexdet]])
        if i % itout[1] == 0:
            # output solution on a grid
            z = u[0:alln]
            solution.append(np.copy(z[ng:-ng:isout]))
            solution_t.append(t)
        i = i + 1
        t = i * dt
        u = RK4(dt, u, rhs, (xs, v, nx, ng))

    solution_det = np.vstack((solution_det))
    solution = np.vstack((solution))

    if return_full:
        return (np.array(solution_t), xs[ng:-ng:isout], solution)
    else:
        return (solution_det[:, 0], solution_det[:, 1], solution_det[:, 2])

def task_solve_pt_plot():

    t, phi, pi = solver(xmin=-300., xmax=300.,
                        nx=401, cfl=0.5, tend=600., itout=(1,10),
                        indexdet=400, stencilorder=4)

    fig, ax1 = plt.subplots()
    ax1.plot(t, phi, '-', color='blue', label='PT')
    ax1.set_xlabel(r'$t$', fontsize='large')
    ax1.set_ylabel(r'$\phi$', fontsize='large')
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
    ax2.semilogy(t, np.fabs(phi), '--', color='red')
    ax2.set_ylabel(r'$|\phi|$', fontsize='large', color="red")
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

    plt.xlim([300, 600])
    fig.tight_layout()
    plt.savefig(FIGPATH+"extracted_at_x402.png")
    plt.show()

def self_convergence():


    tl, low, pi1 = solver(xmin=-300., xmax=300., nx=101, cfl=0.5, tend=600., itout=(1,10),
                        indexdet=50, stencilorder=4, potential=False)
    tm, med, pi2 = solver(xmin=-300., xmax=300., nx=201, cfl=0.5, tend=600., itout=(2,20),
                        indexdet=100, stencilorder=4, potential=False)
    th, hig, pi3 = solver(xmin=-300., xmax=300., nx=401, cfl=0.5, tend=600., itout=(4,10),
                        indexdet=200, stencilorder=4, potential=False)

    # pick solution at detectors x=0.

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
    plt.savefig(FIGPATH+"self_convergence.png",dpi=128)
    plt.show()
    # exit(1)


def find_nearest_index(array, value):
    ''' Finds index of the value in the array that is the closest to the provided one '''
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_solution_at_times():
    list_times = [5., 50., 140., 250.]
    colors = ["blue", "green", "orange", "red"]
    times, xs, sols = solver(xmin=-300., xmax=300.,
                               nx=401, cfl=0.5, tend=600., itout=(1, 5), isout=2,
                               indexdet=400, stencilorder=2,
                               potential=True, return_full=True)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for t, color in zip(list_times, colors):
        idx = find_nearest_index(times, t)
        ax.plot(xs, sols[idx, :], color=color, ls='-', label='t={}'.format(t))

    ax.tick_params(
                axis='both', which='both', labelleft=True,
                labelright=False, tick1On=True, tick2On=True,
                labelsize=int(12),
                direction='in',
                bottom=True, top=True, left=True, right=True
            )

    ax.set_xlabel("$x$", fontsize='large')
    ax.set_ylabel("$\phi$", fontsize='large')
    ax.legend(loc="lower left", ncol=1)
    ax.minorticks_on()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.0)
    plt.savefig(FIGPATH+"solutions_det402_PT.png", dpi=128)
    plt.show()

    #
    # # list_times = [75, 100., 125., 150.]
    # list_times = [150, 175., 200., 225.]
    # # list_times = [75, 100., 125., 150.]
    # list_fnames = ["./wave_rwz/rwz.yg"]
    # colors = ["blue"]
    # lss = ["-"]
    # #
    # res = {}
    # #
    # print("Collecting data")
    #
    #
    #
    # for fname in list_fnames:
    #     res[fname] = {}
    #     times, data = load_solution(fname)
    #     for t in list_times:
    #         res[fname][t] = {}
    #         idx = find_nearest_index(times, t)
    #         x_arr = data[idx, :, 0]
    #         y_arr = data[idx, :, 1]
    #         res[fname][t]["x_arr"] = x_arr
    #         res[fname][t]["y_arr"] = y_arr
    # print("Data collected")
    # #
    # fig = plt.figure(figsize=(len(list_times) * 3.6, 3.6))
    # #
    # axes = []
    # for i in range(1, len(list_times) + 1):
    #     if i == 1:
    #         axes.append(fig.add_subplot(1, len(list_times), i))
    #     else:
    #         axes.append(fig.add_subplot(1, len(list_times), i, sharey=axes[0]))
    # #
    # for ax, t in zip(axes, list_times):
    #     for fname, color in zip(list_fnames, colors):
    #         x_arr = res[fname][t]["x_arr"]
    #         y_arr = res[fname][t]["y_arr"]
    #         ax.plot(x_arr, y_arr, color=color, ls='-', label='{}'.format(fname))
    # #
    # for ax, time in zip(axes, list_times):
    #     ax.set_title("$time:{}$".format(time))
    #     ax.set_xlabel("$x$", fontsize='large')
    #     ax.set_ylabel("$\phi$", fontsize='large')
    #     ax.tick_params(
    #         axis='both', which='both', labelleft=True,
    #         labelright=False, tick1On=True, tick2On=True,
    #         labelsize=int(12),
    #         direction='in',
    #         bottom=True, top=True, left=True, right=True
    #     )
    #     ax.set_xlim(0, x_arr.max())
    #     ax.minorticks_on()
    #     ax.legend(loc="lower left", ncol=1)
    #     # ax.set_ylim(-2.,2.)
    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.2)
    # plt.subplots_adjust(wspace=0.0)
    # plt.savefig("./profiles.png", dpi=128)
    # plt.show()

def plot_solution_3d():
    plot_every = 15

    times, xs, sols = solver(xmin=-300., xmax=300.,
                             nx=401, cfl=0.5, tend=600., itout=(1, 5), isout=2,
                             indexdet=400, stencilorder=2,
                             potential=True, return_full=True)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for t in times[::plot_every]:
        idx = find_nearest_index(times, t)
        z = np.zeros(xs.shape)
        z.fill(times[idx])
        # print("plotting t:{}".format(time))
        ax.plot(xs, sols[idx, :], z, color="blue", ls='-', lw=0.8)

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
    plt.savefig(FIGPATH + "wave_PT_potentials_3d.png", dpi=128)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    task_solve_pt_plot()
    self_convergence()
    plot_solution_at_times()
    plot_solution_3d()