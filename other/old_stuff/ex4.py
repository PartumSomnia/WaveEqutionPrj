#
# Convergence and self-convergence tests for the wave equation implemented
# in ex3.py
#

import numpy as np
import matplotlib.pylab as plt

from ex3 import rhs
from ex3 import rk4

figpath = "/home/vsevolod/GIT/GitLab/projektpraktikum/WaveProject/fig4/"

def find_nearest_index(array, value):
    ''' Finds index of the value in the array that is the closest to the provided one '''
    idx = (np.abs(array - value)).argmin()
    return idx

def func(x, t):
    phi = np.cos(2 * np.pi * (x - t))
    pi = 2 * np.pi * np.sin(2 * np.pi * (x - t)) # d/dt(phi)
    return phi, pi

# def func(x, t):
#     phi = np.cos(2 * np.pi * (x - t)) ** 2
#     pi = 4 * np.pi * np.sin(2 * np.pi * (x - t)) * np.cos(2 * np.pi * (x - t)) # d/dt(phi)
#     return phi, pi

# def func(x, t):
#
#     phi = np.sin(12. * np.pi * (x-t)) ** 4
#     pi = -48. * np.pi * np.cos(12. * np.pi * x) * np.sin(12. * np.pi * (x-t)) ** 3.
#
#     return phi, pi

def solve(N=101, ng=1, xmin=-1., xmax=1., dtdx=0.5,tmax=1.):
    # finide differencing settings
    # N = 101  # physical grid
    # ng = 1  # ghosts
    # xmin = -1
    # xmax = 1
    #
    n = N + 2 * ng  # all points
    dx = (xmax - xmin) / np.float(N - 1)  # spacing
    x = np.arange(start=xmin - dx, stop=xmax + dx + dx, step=dx)
    # x = -dx + xmin + dx * np.arange(n)
    # print(x); exit(1)
    # runge-kutta settings
    dt = dtdx * dx  # stable timestep
    # tmax = 1.
    assert dt < tmax
    res_phi = []
    res_pi = []
    res_t = np.array(np.arange(start=0, stop=tmax, step=dt))
    # initial profile
    phi, pi = func(x, np.zeros(len(x)))
    #
    print("Computing...")
    phi_ = phi  # setting initial profiles
    pi_ = pi  # setting initial profiles
    for i, time in enumerate(res_t):
        if i == 0: # append the 0th solution
            res_phi = np.append(res_phi, phi_[ng:n - ng])
            res_pi = np.append(res_pi, pi_[ng:n - ng])
        else:
            phi_, pi_ = rk4(phi_, pi_, rhs, dt, x, N, ng)
            res_phi = np.append(res_phi, phi_[ng:n - ng])
            res_pi = np.append(res_pi, pi_[ng:n - ng])
        #
        print('{}/{} time: {} res: {} x:{}'.format(i + 1, len(res_t), time, len(phi), len(x)))
    #

    res_phi = np.reshape(res_phi, (len(res_t), len(x[ng:n - ng])))

    print("returning: {} {} {} {}".format(x.shape, res_t.shape, res_phi.shape, res_pi.shape))
    return x[ng:n - ng], res_t, res_phi, res_pi

def plot_1(times, final_x, final_num, final_anal=np.zeros(0,), fname=None):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i, time in enumerate(times):
        if i == 0:
            if len(final_anal):
                ax.scatter(final_x, final_anal[i], color='red', marker='x', label='initial, translated')
            ax.plot(final_x, final_num[i], color='blue', marker='.', label='numerical, computed')
        else:
            if len(final_anal):
                ax.scatter(final_x, final_anal[i], color='red', marker='x')
            ax.plot(final_x, final_num[i], color='blue', marker='.')
    ax.legend(loc='lower left', shadow=False, fontsize='large', frameon=False)
    # ax.set_yscale("log")
    ax.set_xlabel("x", fontsize='large')
    # ax.set_ylabel()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.minorticks_on()
    ax.set_title("1 period of numerical vs. analytical (translated)")
    ax.legend(loc="lower left", ncol=1)
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(figpath + fname, dpi=128)
    plt.tight_layout()
    plt.show()

def convergence(anal_phi1, anal_phi2, num_phi1, num_phi2):

    #
    anal_phi2 = anal_phi2[::2]
    num_phi2 = num_phi2[::2]
    #
    diffs1 = []
    diffs2 = []
    norms = []
    for i, time in enumerate(anal_phi1):
        #
        print("time:{} num_phi1:{} num_phi2:{}"
              .format(time, len(num_phi1[i]), len(num_phi2[i])))
        assert int(2 * len(num_phi1[i]) - 1) == len(num_phi2[i])
        #

        #
        sol_anal1 = anal_phi1[i]
        sol_anal2 = anal_phi2[i]
        sol_num = num_phi1[i]
        sol_num_2 = num_phi2[i]
        #
        diff1 = np.abs(sol_anal1 - sol_num)  # same length
        diff2 = np.abs(sol_anal2 - sol_num_2)[::2]  # same length
        print("diff1: {:d} diff2: {:d}".format(len(diffs1), len(diffs2)))
        #
        norm2 = np.log2(np.sqrt(np.sum(diff1 ** 2)) / np.sqrt(np.sum(np.abs(diff2 ** 2))))
        #
        diffs1.append(diff1)
        diffs2.append(diff2)
        norms.append(norm2)
    return diffs1, diffs2, norms

def get_anal(x_arr, t_arr):
    anal_final = []
    phi = []
    for i, t in enumerate(t_arr):
        _t = np.zeros(len(x_arr))
        _t.fill(t)
        phi, _ = func(x_arr, _t)
        anal_final.append(phi)
    anal_final = np.reshape(anal_final, (len(t_arr), len(x_arr)))
    return anal_final

def plot_convergence(times, x_arr, diffs1, diffs2, norms, fname="convergence.png"):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(times, norms, color='blue', marker='.', label='$\log_2(norm2)$')

    ax.legend(loc='lower left', shadow=False, fontsize='large', frameon=False)
    # ax.set_yscale("log")
    ax.set_xlabel("time", fontsize='large')
    # ax.set_ylabel()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.minorticks_on()
    ax.set_title("Convergence Study")
    ax.legend(loc="lower left", ncol=1)
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(figpath + fname, dpi=128)
    plt.tight_layout()
    plt.show()

def self_convergence(num_phi1, num_phi2, num_phi4):

    print(len(num_phi1), len(num_phi2), len(num_phi4))

    num_phi2 = num_phi2[::2]
    num_phi4 = num_phi4[::4]
    diffs1 = []
    diffs2 = []
    norms = []
    for i in range(len(num_phi1)):
        sol_num = num_phi1[i]
        sol_num_2 = num_phi2[i]
        sol_num_4 = num_phi4[i]
        #
        diff1 = np.abs(sol_num - sol_num_2[::2])  # same length
        diff2 = np.abs(sol_num_2 - sol_num_4[::2])[::2]  # same length
        print("self_conv diff1: {:d} diff2: {:d}".format(len(diff1), len(diff2)))
        #
        norm2 = np.log2(np.sqrt(np.sum(diff1 ** 2)) / np.sqrt(np.sum(np.abs(diff2 ** 2))))
        #
        diffs1.append(diff1)
        diffs2.append(diff2)
        norms.append(norm2)
    return diffs1, diffs2, norms


''' solutions at varius dxdt compare '''
# if __name__ == '__main__':
#     list_times = [0., 5., 10.]
#     list_dtdx = [0.5, 1., 1.5]
#     colors = ["blue", "orange", "red"]
#
#     list_resolutions = [101]
#     lss = ["-"]
#     #
#     res = {}
#     #
#     for dxdt in list_dtdx:
#         res[dxdt] = {}
#         for npoints in list_resolutions:
#             res[dxdt][npoints] = {}
#             x_arr, res_t, res_phi, _ = solve(N=npoints, ng=1, xmin=0., xmax=10., dtdx=dxdt, tmax=10.0)
#             anal_final1 = get_anal(x_arr, res_t)
#             res[dxdt][npoints]["x_arr"] = x_arr
#             res[dxdt][npoints]["res_t"] = res_t
#             res[dxdt][npoints]["res_phi"] = res_phi
#             res[dxdt][npoints]["phi_anal"] = anal_final1
#     print("computed")
#     #
#     fig, axes = plt.subplots(nrows=1, ncols=len(list_times), sharey='all', figsize=(12.6, 4.2))
#     if len(list_times) == 1:
#         axes = [axes]
#     #
#     for dxdt, color in zip(list_dtdx, colors):
#         for npoints, ls in zip(list_resolutions, lss):
#             #
#             x_arr = res[dxdt][npoints]["x_arr"]
#             res_t = res[dxdt][npoints]["res_t"]
#             res_phi = res[dxdt][npoints]["res_phi"]
#             phi_anal = res[dxdt][npoints]["phi_anal"]
#             #
#             for ax, time in zip(axes, list_times):
#                 #
#                 idx = find_nearest_index(res_t, time)
#                 x = x_arr
#                 phi = res_phi[idx]
#                 phi_a = phi_anal[idx]
#                 #
#                 ax.plot(x, phi, color=color, ls=ls, label='n:{} dt/dx:{}'.format(npoints, dxdt))
#                 ax.plot(x, phi_a, color='gray', ls=':')#, label='n:{} dt/dx:{}'.format(npoints, dxdt))
#     #
#     for ax, time in zip(axes, list_times):
#         ax.set_title("$time:{}$".format(time))
#         ax.set_xlabel("$x$", fontsize='large')
#         ax.set_ylabel("$\phi$", fontsize='large')
#         ax.tick_params(
#             axis='both', which='both', labelleft=True,
#             labelright=False, tick1On=True, tick2On=True,
#             labelsize=int(12),
#             direction='in',
#             bottom=True, top=True, left=True, right=True
#         )
#         ax.minorticks_on()
#         ax.legend(loc="lower left", ncol=1)
#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.2)
#     plt.subplots_adjust(wspace=0.0)
#     plt.savefig(figpath + "profiles.png", dpi=128)
#     plt.show()

# exit(1)





''' self-convergence test many dtdx'''
# if __name__ == '__main__':
#     # solution at one reoslution
#     list_dtdx = [0.5, 1., 1.5]
#     colors = ["blue", "orange", "red"]
#     time_arrays = []
#     norm2_arrays = []
#     for dxdt in list_dtdx:
#         x_arr, res_t, res_phi, _ = solve(N=101, ng=1, xmin=0., xmax=5., dtdx=dxdt, tmax=10.0)
#         x_arr2, res_t2, res_phi2, _ = solve(N=201, ng=1, xmin=0., xmax=5., dtdx=dxdt, tmax=10.0)
#         x_arr4, res_t4, res_phi4, _ = solve(N=401, ng=1, xmin=0., xmax=5., dtdx=dxdt, tmax=10.0)
#         # print(res_t[:10], res_t[-1])
#         # print(res_t2[::2][:10], res_t2[-1])
#         # print(len(res_t), len(res_t2[::2])); exit(1)
#         # times, final_x, final_anal, final_num = get_num_anal_solutions(x, res_t, res_phi)
#         # print("len(phi):{}".format(len(final_num[0])))
#         #---------------------
#
#         # --------------------
#         # plot_1(t_arr, x_arr, res_phi, anal_final, fname='plot1.png')
#         # --------------------
#         diffs1, diffs2, norms = self_convergence(res_phi, res_phi2, res_phi4)
#         # plot_convergence(res_t, x_arr, diffs1, diffs2, norms)
#         #
#         time_arrays.append(res_t)
#         norm2_arrays.append(norms)
#     # plotting
#     fig, ax = plt.subplots(nrows=1, ncols=1)
#     for time_arr, norm2, color, dxdt in zip(time_arrays, norm2_arrays, colors, list_dtdx):
#         ax.plot(time_arr, norm2, color=color, ls='-', label='dxdt:{}'.format(dxdt))
#
#     # ax.set_yscale("log")
#     ax.set_xlabel("time", fontsize='large')
#     # ax.set_ylabel()
#     ax.tick_params(
#         axis='both', which='both', labelleft=True,
#         labelright=False, tick1On=True, tick2On=True,
#         labelsize=int(12),
#         direction='in',
#         bottom=True, top=True, left=True, right=True
#     )
#     ax.set_ylabel("$\log_2(norm2)$", fontsize='large')
#     ax.minorticks_on()
#     ax.set_title("Convergence Study")
#     ax.set_ylim(0, 2.1)
#     ax.legend(loc="lower left", ncol=1)
#     plt.subplots_adjust(hspace=0.2)
#     plt.subplots_adjust(wspace=0.2)
#     plt.savefig(figpath + "self_convergence_dxdt.png", dpi=128)
#     plt.tight_layout()
#     plt.show()
#     print("me")
# exit(1)


''' convergence test many dtdx'''
if __name__ == '__main__':
    # solution at one reoslution
    list_dtdx = [0.5, 1., 1.5]
    colors = ["blue", "orange", "red"]
    time_arrays = []
    norm2_arrays = []
    for dxdt in list_dtdx:
        x_arr, res_t, res_phi, _ = solve(N=101, ng=1, xmin=0., xmax=5., dtdx=dxdt, tmax=10.0)
        x_arr2, res_t2, res_phi2, _ = solve(N=201, ng=1, xmin=0., xmax=5., dtdx=dxdt, tmax=10.0)
        # print(res_t[:10], res_t[-1])
        # print(res_t2[::2][:10], res_t2[-1])
        # print(len(res_t), len(res_t2[::2])); exit(1)
        # times, final_x, final_anal, final_num = get_num_anal_solutions(x, res_t, res_phi)
        # print("len(phi):{}".format(len(final_num[0])))
        #---------------------
        anal_final1 = get_anal(x_arr, res_t)
        anal_final2 = get_anal(x_arr2, res_t2)
        # --------------------
        # plot_1(t_arr, x_arr, res_phi, anal_final, fname='plot1.png')
        # --------------------
        diffs1, diffs2, norms = convergence(anal_final1, anal_final2, res_phi, res_phi2)
        # plot_convergence(res_t, x_arr, diffs1, diffs2, norms)
        #
        time_arrays.append(res_t)
        norm2_arrays.append(norms)
    # plotting
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for time_arr, norm2, color, dxdt in zip(time_arrays, norm2_arrays, colors, list_dtdx):
        ax.plot(time_arr, norm2, color=color, ls='-', label='dxdt:{}'.format(dxdt))

    # ax.set_yscale("log")
    ax.set_xlabel("time", fontsize='large')
    # ax.set_ylabel()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.set_ylabel("$\log_2(norm2)$", fontsize='large')
    ax.minorticks_on()
    ax.set_title("Convergence Study")
    ax.legend(loc="lower left", ncol=1)
    ax.set_ylim(0, 2.1)
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(figpath + "convergence_dxdt.png", dpi=128)
    plt.tight_layout()
    print("me")
    plt.show()

exit(1)


''' self-convergence '''
if __name__ == '__main__':
    # solution at one reoslution
    x_arr, res_t, res_phi, _ = solve(N=101, ng=1, xmin=0., xmax=2., dtdx=1., tmax=5.0)
    x_arr2, res_t2, res_phi2, _ = solve(N=201, ng=1, xmin=0., xmax=2., dtdx=1., tmax=5.0)
    x_arr4, res_t4, res_phi4, _ = solve(N=401, ng=1, xmin=0., xmax=2., dtdx=1., tmax=5.0)

    # --------------------
    # plot_1(t_arr, x_arr, res_phi, anal_final, fname='plot1.png')
    # --------------------
    diffs1, diffs2, norms = self_convergence(res_phi, res_phi2, res_phi4)
    plot_convergence(res_t, x_arr, diffs1, diffs2, norms, fname="self_convergence.png")
exit(1)


''' convergence test'''
if __name__ == '__main__':
    # solution at one reoslution
    x_arr, res_t, res_phi, _ = solve(N=101, ng=1, xmin=0., xmax=2., dtdx=1.0, tmax=5.0)
    x_arr2, res_t2, res_phi2, _ = solve(N=201, ng=1, xmin=0., xmax=2., dtdx=1.0, tmax=5.0)
    # print(res_t[:10], res_t[-1])
    # print(res_t2[::2][:10], res_t2[-1])
    # print(len(res_t), len(res_t2[::2])); exit(1)
    # times, final_x, final_anal, final_num = get_num_anal_solutions(x, res_t, res_phi)
    # print("len(phi):{}".format(len(final_num[0])))
    #---------------------
    anal_final1 = get_anal(x_arr, res_t)
    anal_final2 = get_anal(x_arr2, res_t2)
    # --------------------
    # plot_1(t_arr, x_arr, res_phi, anal_final, fname='plot1.png')
    # --------------------
    diffs1, diffs2, norms = convergence(anal_final1, anal_final2, res_phi, res_phi2)
    plot_convergence(res_t, x_arr, diffs1, diffs2, norms)
exit(1)





''' ----------- '''





def find_nearest_index(array, value):
    ''' Finds index of the value in the array that is the closest to the provided one '''
    idx = (np.abs(array - value)).argmin()
    return idx

def get_one_period(x, phi, t=1.):
    #
    x = np.array(x)
    phi = np.array(phi)
    phi_max = np.max(phi)

    i_max = find_nearest_index(phi, phi.max())
    x1 = x[i_max]
    x2 = x[i_max] + t
    idx1 = find_nearest_index(x, x1)
    idx2 = find_nearest_index(x, x2)
    # i_max2 = np.where((phi == phi.max()) & (phi != phi[i_max1]))
    # print(); exit(1)
    #
    x = x[idx1:idx2]
    phi = phi[idx1:idx2]
    #
    return x, phi

def init_prof(x):

    phi = np.cos(2 * np.pi  * x)
    pi = 2 * np.pi * np.sin(2 * np.pi * x)

    # phi = np.sin(2 * np.pi  * x)
    # pi = 2 * np.pi * np.cos(2 * np.pi * x)

    return phi, pi

#
def solve(N=101, ng=1, xmin=-1., xmax=1., dtdx=0.5,tmax=1.):
    # finide differencing settings
    # N = 101  # physical grid
    # ng = 1  # ghosts
    # xmin = -1
    # xmax = 1
    #
    n = N + 2 * ng  # all points
    dx = (xmax - xmin) / np.float(N - 1)  # spacing
    x = np.arange(start=xmin - dx, stop=xmax + dx + dx, step=dx)
    # runge-kutta settings
    dt = dtdx * dx  # stable timestep
    # tmax = 1.
    assert dt < tmax
    res_phi = []
    res_pi = []
    res_t = np.array(np.arange(start=0, stop=tmax, step=dt))
    #
    phi, pi = init_prof(x)
    #
    print("Computing...")
    phi_ = phi  # setting initial profiles
    pi_ = pi  # setting initial profiles
    for i in range(int(tmax / dt)):
        phi_, pi_ = rk4(phi_, pi_, rhs, dt, x, N, ng)
        res_phi = np.append(res_phi, phi_[ng:n - ng])
        res_pi = np.append(res_pi, pi_[ng:n - ng])
        #
        print('{}/{} res: {} x:{}'.format(i + 1, int(tmax / dt), len(phi), len(x)))
    #
    res_phi = np.reshape(res_phi, (int(tmax / dt), len(x[ng:n - ng])))

    print("returning: {} {} {} {}".format(x.shape, res_t.shape, res_phi.shape, res_pi.shape))
    return x[ng:n - ng], res_t, res_phi, res_pi

def get_num_anal_solutions(x, res_t, res_phi, period=1):
    #
    print('shape(res_phi): {} '.format(res_phi.shape))
    # print('shape(res_pi): {} '.format(res_pi.shape))
    #
    print("selecting 1 period...")
    times = []
    final_x = []
    final_num = []
    final_anal = []
    # select one period from iniial profile
    phi, pi = init_prof(x)
    x1, phi1 = get_one_period(x, phi, t=period)
    # anal solution (1 period)
    tmp, _ = init_prof(x)
    i_max = np.argmax(tmp)
    tmp_plot = tmp[i_max:i_max + len(x1)]
    tmp_x = x[i_max:i_max + len(x1)]
    # init_i_min = np.argmin(tmp_plot)
    # init_x_min = tmp_x[init_i_min]
    # ax.plot(tmp_x, tmp_plot, color='red', ls='-')
    # ax.plot(init_x_min, tmp[init_i_min], color='red', marker='x', ms=7.)
    #
    final_anal = tmp_plot
    #
    for i in range(len(res_t)):
        if i % 20 == 0:
            i_max = find_nearest_index(res_phi[i], phi1.max())
            print(i_max)
            x_plot = x[i_max:i_max+len(x1)]
            phi_plot = res_phi[i, :][i_max:i_max+len(x1)]
            # ax.plot(x_plot, phi_plot, color='black', ls='-', lw=0.8)  # , label=r'$\phi_0$')

            #
            # i_min1 = np.argmin(phi_plot)
            # print("min",i_min1)
            # ax.plot(x_plot[i_min1], phi_plot[i_min1], color='blue', marker='o', ms=7.)
            #
            # tmp_x_ = tmp_x + (x_plot[i_min1] - init_x_min)
            # print(x_plot[i_min1], init_x_min, i_min1)
            #

            # ax.scatter(tmp_x_, tmp_plot, color='gray', marker='x')
            print("time:{} i_max:{}, tmp_x:{}, x_plot:{}".format(res_t[i],i_max, len(tmp_x), len(x_plot)))
            times.append(res_t[i])
            final_x.append(x_plot)
            final_num.append(phi_plot)
            # final_anal.append(tmp_plot)

    print("returning: {} {} {} {}".format(len(times), len(final_x), len(final_anal),  len(final_num)))
    return times, final_x, final_anal, final_num

def plot_1(times, final_x, final_anal, final_num, fname):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i, time in enumerate(times):
        if i == 0:
            # ax.scatter(final_x[i], final_anal, color='red', marker='x', label='initial, translated')
            ax.plot(final_x[i], final_num[i], color='blue', marker='.', label='numerical, computed')
        else:
            # ax.scatter(final_x[i], final_anal, color='red', marker='x')
            ax.plot(final_x[i], final_num[i], color='blue', marker='.')
    ax.legend(loc='lower left', shadow=False, fontsize='large', frameon=False)
    # ax.set_yscale("log")
    ax.set_xlabel("x", fontsize='large')
    # ax.set_ylabel()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.minorticks_on()
    ax.set_title("1 period of numerical vs. analytical (translated)")
    ax.legend(loc="lower left", ncol=1)
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(figpath + fname, dpi=128)
    plt.tight_layout()
    plt.show()

def convergence(res_t, x1, anal_phi1, num_phi1, x2, anal_phi2, num_phi2):
    sol_anal = anal_phi1
    sol_num = num_phi1
    sol_num_2 = num_phi2
    #
    diffs1 = []
    diffs2 = []
    norms = []
    for i, time in enumerate(times):
        #
        print("time:{} num_phi1:{} num_phi2:{}"
              .format(time, len(num_phi1[i]), len(num_phi2[i])))
        assert 2 * len(num_phi1[i]) == len(num_phi2[i])
        #
        sol_anal = anal_phi1
        sol_num = num_phi1[i]
        sol_num_2 = num_phi2[i]
        #
        diff1 = np.abs(sol_anal - sol_num)  # same length
        diff2 = np.abs(sol_anal - sol_num_2[::2])  # same length
        #
        norm2 = np.sqrt(np.sum(diff1 ** 2)) / np.sqrt(np.sum(np.abs(diff2 ** 2)))
        #
        diffs1.append(diff1)
        diffs2.append(diff2)
        norms.append(norm2)
    return times, diffs1, diffs2, norms




''' Convergence test '''
if __name__ == '__main__':
    # solution at one reoslution
    x_arr, t_arr, res_phi, _ = solve(N=201, ng=1, xmin=0., xmax=2., dtdx=.5, tmax=1.)
    times, final_x, final_anal, final_num = get_num_anal_solutions(x_arr, t_arr, res_phi)
    print("len(phi):{}".format(len(final_num[0])))
    #plot_1(times, final_x, final_anal, final_num, fname='plot1.png')
    #
    x2, t_arr, res_phi2, _ = solve(N=401, ng=1, xmin=0, xmax=2., dtdx=.5, tmax=1.)
    _, final_x2, final_anal2, final_num2 = get_num_anal_solutions(x2, t_arr, res_phi2)
    print("len(phi2):{}".format(len(final_num2[0])))
    #plot_1(times, final_x2, final_anal2, final_num2, fname='plot2.png')
    convergence(times, final_x, final_anal, final_num,
                final_x2, final_anal2, final_num2)


exit(1)

if __name__ == '__main__':
    # --------- setup --------
    N = 101     # physical grid
    ng = 1      # ghosts
    xmin = -1
    xmax = 1
    # -----------------------

    n = N + 2 * ng # all points
    dx = (xmax - xmin) / np.float(N - 1) # spacing
    x_arr = np.arange(start =xmin - dx, stop =xmax + dx + dx, step = dx) # from xmin-dx to xmax+dx

    print('------- grid -------')
    print(x_arr)
    print('--------------------')

    # solution is to be stored
    # phi = np.zeros_like(x) #
    # pi = np.zeros_like(x) # pi = d/dt(phi)

    # inital profile
    phi, pi = init_prof(x_arr)
    #
    x1, phi1 = get_one_period(x_arr, phi)
    print("One period, max to max, span: {} points".format(len(x1)))
    #
    # ------------------------------------ 1d ------------------------------

    dt = 1.0 * dx  # stable timestep
    tmax = 1.
    assert dt < tmax
    res_phi = []
    res_pi = []
    t_arr = np.array(np.arange(start=0, stop=tmax, step=dt))
    # print(dt, tmax, res_t); exit(1)
    print("Computing...")
    phi_ = phi  # setting initial profiles
    pi_ = pi    # setting initial profiles
    for i in range(int(tmax / dt)):
        phi_, pi_ = rk4(phi_, pi_, rhs, dt, x_arr, N, ng)
        res_phi = np.append(res_phi, phi_)
        res_pi = np.append(res_pi, pi_)
        #
        print('{}/{} res: {} x:{}'.format(i + 1, int(tmax / dt), len(phi), len(x_arr)))
    res_phi = np.reshape(res_phi, (int(tmax / dt), len(x_arr)))
    # res_pi = np.reshape(res_pi, (int(tmax / dt), len(x)))
    print("res_phi: {}".format(res_phi.shape))
    # print("res_pi:  {}".format(res_pi.shape))
    print("res_t:  {}".format(t_arr.shape))
    #
    # plot initial profile
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(x[ng:n - ng], phi[ng:n - ng], color='red', ls='-', lw=1., label=r'$\phi_0$')
    #
    tmp, _ = init_prof(x_arr[ng:n - ng])
    i_max = np.argmax(tmp)
    tmp_plot = tmp[i_max:i_max + len(x1)]
    tmp_x = x_arr[ng:n - ng][i_max:i_max + len(x1)]
    init_i_min = np.argmin(tmp_plot)
    init_x_min = tmp_x[init_i_min]
    ax.plot(tmp_x, tmp_plot, color='red', ls='-')
    ax.plot(init_x_min, tmp[init_i_min], color='red', marker='x', ms=7.)
    #
    #
    for i in range(len(t_arr)):
        if i % 20 == 0:
            i_max = np.argmax(res_phi[i, ng:n - ng])
            print(i_max)
            x_plot = x_arr[ng:n - ng][i_max:i_max + len(x1)]
            phi_plot = res_phi[i, ng:n - ng][i_max:i_max+len(x1)]
            ax.plot(x_plot, phi_plot, color='black', ls='-', lw=0.8)  # , label=r'$\phi_0$')
            # init_plot = tmp[i_max:i_max+len(x1)]
            # assert len(init_plot) == len(x_plot)
            #
            i_min1 = np.argmin(phi_plot)
            print("min",i_min1)
            ax.plot(x_plot[i_min1], phi_plot[i_min1], color='blue', marker='o', ms=7.)
            #
            # tmp_x_ = tmp_x + (x_plot[i_min1] - init_x_min)
            # print(x_plot[i_min1], init_x_min, i_min1)
            #

            ax.scatter(x_plot, tmp_plot, color='gray', marker='x')
            #
            # print(tmp_x_)
            # print(x_plot)
            # print('----------')

    ax.legend(loc='lower left', shadow=False, fontsize='large', frameon=False)
    # ax.set_yscale("log")
    ax.set_xlabel("x", fontsize='large')
    # ax.set_ylabel()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.minorticks_on()
    ax.set_title("Initial profile")
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(figpath + "conv1d.png", dpi=128)
    plt.tight_layout()
    plt.show()
