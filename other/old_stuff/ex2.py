import numpy as np
import matplotlib.pyplot as plt

#
#
#
#
#
#
#
figpath = "/home/vsevolod/GIT/GitLab/projektpraktikum/WaveProject/fig2/"
# def deriv2(y, t, omega=1):
#     ret
from docutils.nodes import table


def deriv(y, t, omega=1.):
    '''
        computes a derivative of a state vector y
        derivative of a position y[0] is velocity -- y[1]
        derivative of the velocity -- acceleration of a point mass -omega * y[0]
    '''
    xdot = y[1]              # velcoity
    ydot = -omega * y[0]    # x
    return np.array([xdot, ydot])

def rk4(y, f, t, h):
    k1 = f(y, t)
    k2 = f(y + h / 2 * k1, t + h / 2)
    k3 = f(y + h / 2 * k2, t + h / 2)
    k4 = f(y + h * k3, t + h)
    y = y + h * (k1 + 2 * k2 + 2 * k3  + k4) / 6
    t = t + h
    return (t, y)

def anal(t, x0, p0, m=1., omega=1.):
    """
    Analytical solution to the harmonic oscillator equation.
    See eq. 60 in the solution.tex
    :param t:
    :param p0:
    :param x0:
    :param m:
    :param omega:
    :return:
    """
    t = np.array(t, dtype=float)
    # x1 = (-p0/(2 * m * np.sqrt(-1*omega)) + x0/2) * np.exp(-t * np.sqrt(-1 * omega))
    # x2 = (p0/(2 * m * np.sqrt(-1*omega)) + x0/2) * np.exp(t * np.sqrt(-1 * omega))

    x1 = (-p0/(2 * m * 1j * np.sqrt(omega)) + x0/2) * np.exp(-t * 1j * np.sqrt(omega))
    x2 = (p0/(2 * m * 1j * np.sqrt(omega)) + x0/2) * np.exp(t * 1j * np.sqrt(omega))

    p1 = - 1j * np.sqrt(omega) * (-p0/(2 * m * 1j * np.sqrt(omega)) + x0/2) * np.exp(-t * 1j * np.sqrt(omega))
    p2 =  1j * np.sqrt(omega) * (p0/(2 * m * 1j * np.sqrt(omega)) + x0/2) * np.exp(t * 1j * np.sqrt(omega))

    return x1 + x2, p1 + p2

def solve_harm_oscill(x0, p0, t0, tmax, h):

    t = t0                  # initializing time to t0
    y = np.array([x0, p0])  # initializing state-vector, consisting of evolving variables
    #
    ts = np.array([t])  # to store all times
    ys = np.array([y])  # to store all solution points

    system = deriv # system of equations to evolve (depends on state-vector and time)

    for i in range(int(tmax / h)):
        (t, y) = rk4(y, system, t, h)
        ts = np.append(ts, t)
        # print(y, ys.shape)
        ys = np.concatenate((ys, np.array([y])))

    [y_anmplitude, y_velocity] = ys.transpose()

    print("h:{} N(ts): {}".format(h, len(ts)))

    return ts, y_anmplitude, y_velocity

def convergence(time_arr, timee_arr2, sol_anal, sol_num, sol_num_2):
    assert len(time_arr) == len(sol_anal)
    assert len(time_arr) == len(sol_num)
    assert len(timee_arr2) == len(sol_num_2)
    #
    h = np.diff(time_arr)
    h2 = np.diff(timee_arr2)
    print("h:{} [{}] h2:{} [{}]".format(h, h2, len(time_arr), len(timee_arr2)))

    # print(timee_arr2[::2])
    # print(sol_num_2[::2])
    #
    # exit(1)
    diff1 = np.abs(sol_anal - sol_num) # same length
    diff2 = np.abs(sol_anal - sol_num_2[::2]) # same length
    #
    norm2 = np.sqrt(np.sum(diff1 ** 2)) / np.sqrt(np.sum(np.abs(diff2 ** 2)))
    #

    fig, ax, = plt.subplots(ncols=1, nrows=1, figsize=(4.2,3.6))
    # ax.set_aspect('equal')
    ax.set_title(r"Convergence", fontsize='large')
    ax.plot(time_arr, diff1, marker='.', color='black', label=r"$|f_{E}' - f_{h}'|$")
    ax.plot(time_arr, diff2, ls='-', lw=0.8, color='gray', label=r"$|f_{E}' - f_{h/2}'|$")
    ax.legend(loc='lower right', shadow=False, fontsize='large', frameon=False)
    ax.set_yscale("log")
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
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(figpath + "convergence_harm_osc.png", dpi=128)
    plt.tight_layout()
    plt.show()
    # plt.close()

    #
    return norm2

def selfconvergence(time_arr1, time_arr2, sol_num, sol_num_2, sol_num_4):

    #
    #
    # print("h:{} [{}] h2:{} [{}]".format(h, h2, len(time_arr), len(timee_arr2)))

    #
    # exit(1)
    diff1 = np.abs(sol_num - sol_num_2[::2]) # same length
    diff2 = np.abs(sol_num_2 - sol_num_4[::2]) # same length
    #
    norm2 = np.sqrt(np.sum(diff1 ** 2)) / np.sqrt(np.sum(np.abs(diff2 ** 2)))

    fig, ax, = plt.subplots(ncols=1, nrows=1, figsize=(4.2, 3.6))
    # ax.set_aspect('equal')
    ax.set_title(r"Self-Convergence", fontsize='large')
    ax.plot(time_arr1, diff1, marker='.', color='black', label=r"$|f_{h}' - f_{h/2}'|$")
    ax.plot(time_arr2, diff2, ls='-', lw=0.8, color='gray', label=r"$|f_{h/2}' - f_{h/4}'|$")
    ax.legend(loc='lower right', shadow=False, fontsize='large', frameon=False)
    ax.set_yscale("log")
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
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(figpath + "self-convergence_harm_osc.png", dpi=128)
    plt.tight_layout()
    plt.show()



    return norm2

if __name__ == '__main__':
    accuracty = 4 # if the 4th runge-cutta is used
    # time resolution 1
    ts, y_anmplitude, y_velocity = solve_harm_oscill(x0=0, p0=1., t0=0.,tmax=20., h=0.2)
    enegry_num1 = (y_anmplitude ** 2 + y_velocity ** 2)

    anal_amplitude, anal_velcity = anal(ts, x0=0, p0=1., m=1., omega=1.)
    enegry_anal = (anal_velcity ** 2 + anal_amplitude ** 2)

    #
    ts2, y_anmplitude2, y_velocity2 = solve_harm_oscill(x0=0, p0=1., t0=0.,tmax=20., h=0.1)
    enegry_num2 = (y_anmplitude2 ** 2 + y_velocity2 ** 2)
    # anal_amplitude2, anal_velcity2 = anal(ts, x0=0, p0=1., m=1., omega=1.)
    ts4, y_anmplitude4, y_velocity4 = solve_harm_oscill(x0=0, p0=1., t0=0., tmax=20., h=0.05)
    enegry_num4 = (y_anmplitude4 ** 2 + y_velocity4 ** 2)
    # print(ts)
    # print(ts2[::2])
    #
    print("E_anal:{}".format(enegry_anal))
    print("E_num:{}".format(enegry_num1))
    print("E_num2:{}".format(enegry_num2))
    print("E_num4:{}".format(enegry_num4))

    theta = convergence(time_arr=ts, timee_arr2=ts2, sol_anal=anal_amplitude, sol_num=y_anmplitude, sol_num_2=y_anmplitude2)
    print("Convergence: {} Expected(accuracy): {}".format(theta, 2**float(accuracty)))
    #
    theta2 = selfconvergence(time_arr1=ts, time_arr2=ts2, sol_num=y_anmplitude, sol_num_2=y_anmplitude2, sol_num_4=y_anmplitude4)
    print("Self-Convergence: {} Expected(accuracy): {}".format(theta2, 2**float(accuracty)))
    #


    fig, ax, = plt.subplots(ncols=1, nrows=1, figsize=(4.2, 3.6))
    # ax.set_aspect('equal')
    ax.set_title(r"Energy Conservation", fontsize='large')
    ax.plot(ts, enegry_anal, ls='-', color='black', label=r"anal E")
    ax.plot(ts, enegry_num1, ls='-', lw=0.8, color='green', label=r"num E $h$")
    ax.plot(ts2, enegry_num2, ls='-', lw=0.8, color='blue', label=r"num E $h/2$")
    ax.plot(ts4, enegry_num4, ls='-', lw=0.8, color='red', label=r"num E $h/4$")
    ax.legend(loc='lower left', shadow=False, fontsize='large', frameon=False)
    ax.set_yscale("log")
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
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(figpath + "energy_conservation.png", dpi=128)
    plt.tight_layout()
    plt.show()




    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7.8, 3.6))

    ax = axes[0]
    ax.set_title(r"Equation $x_{tt} = -x$", fontsize='large')
    ax.plot(ts,y_anmplitude,marker='.', color="blue", label='amplitude', alpha=0.7)
    ax.plot(ts,y_velocity,marker='.',color="red", label='velocity', alpha=0.7)
    ax.plot(ts, anal_amplitude, ls='-',lw=0.8, color='blue',label="anal amp")
    ax.plot(ts, anal_velcity, ls='-', lw=0.8, color='red', label="anal vel")
    # ax.plot(ts, y2,marker='.',label='acceleration')
    ax.legend(loc='lower left', shadow=False, fontsize='large', frameon=True)
    ax.set_xlabel("time", fontsize='x-large')
    ax.set_ylabel(r'state-vector $y=[x, \dot{x}]$', fontsize='x-large')
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.minorticks_on()
    #
    ax = axes[1]
    # ax.set_aspect('equal')
    ax.set_title(r"$E = x^2 + p^2 = {:.2f}$".format(anal_velcity[0] ** 2 + anal_amplitude[0] ** 2), fontsize='large')
    ax.plot(y_anmplitude, y_velocity, marker='.', color='black', label='Num')
    ax.plot(anal_amplitude, anal_velcity, ls='-', lw=0.8, color='gray', label='Anal')
    ax.legend(loc='lower left', shadow=False, fontsize='large', frameon=True)
    ax.set_xlabel("$x$", fontsize='large')
    ax.set_ylabel("$\dot{x}$", fontsize='large')
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.minorticks_on()
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(figpath + "harmonic_oscillator.png", dpi=128)
    plt.tight_layout()
    plt.show()



    #

