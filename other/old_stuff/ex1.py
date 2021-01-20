#
# Implementation of the finitie differencing
# function \texttt{findif(f, n, x1, x2, ord, acc, method)}, that performs the finite differencing of a function $f(x)$,
# using $n$ equally spaced points between $x1$ and $x2$, using the stencil of order $ord$ and accuracy $acc$ with method
# $method$. \\
# Possible orders: $[1,2]$ for the first and second order derivative \\
# Possible accuracies: $[2,4]$ for the second and 4th order accuracy stencils \\
# Possible methods: $ghost$, $onesided$. The first one will extend the $x$ to 1 (2) points in left and right for second
# (forth) accuracies and computes the central finite differencing for points between $0$ to $n$, using added points as
# ghosts. Second method uses one-sided stencils for 1 (2) border points, where central stencil cannot be used. \\
# To write stencils the coefficients from \url{https://en.wikipedia.org/wiki/Finite_difference_coefficient} were used
#


import numpy as np
import matplotlib.pyplot as plt




def f1(x):
    return (x - 0.5)**2. + x
def df1(x):
    return 2. * x
def ddf1(x):
    res = np.zeros(len(x))
    res.fill(2.)
    return res

def f2(x):
    return (x - 0.5)**3. + (x - 0.5)**2. + x
def df2(x):
    return 3. * x ** 2. - x + (3./4.)
def ddf2(x):
    return 6. * x - 1.

def f3(x):
    return np.sqrt(x)
def df3(x):
    return 1. / (2. * np.sqrt(x))
def ddf3(x):
    return -1. / (4. * x ** (3./2.))

def f4(x):
    return np.sin(12. * np.pi * x)
def df4(x):
    return 12 * np.pi * np.cos(12. * np.pi * x)
def ddf4(x):
    return -144. * np.pi ** 2 * np.sin(12. * np.pi * x)

def f5(x):
    return np.sin(12. * np.pi * x) ** 4.
def df5(x):
    return 48. * np.pi * np.cos(12. * np.pi * x) * np.sin(12. * np.pi * x) ** 3.
def ddf5(x):
    return -576. * np.pi ** 2. * np.sin(12. * np.pi * x) ** 2. * (-3 * np.cos(12 * np.pi * x) ** 2. + np.sin(12 * np.pi * x) ** 2.)


def f6(x, a=5.):
    return np.exp(-a * x ** 2.)
def df6(x, a=5.):
    return -2. * a * np.exp(-a * x ** 2.) * x
def ddf6(x, a=5.):
    return 2. * a * np.exp( - a * x ** 2.) * (-1. + 2. * a * x ** 2.)

# differencing
# def left_sided_second_order(i, func, step):
#     return (-3 * func[i] + 4 * func[i + 1] - func[i + 2]) / (2 * step)
#
# def right_sided_second_order(i, func, step):
#     return (3 * func[i] - 4 * func[i - 1] + func[i - 2]) / (2 * step)
#
#
#
# def central_second_order(i, func, step):
#     return (func[i + 1] - func[i - 1]) / (2 * step)
#
# def central_koeff(derivative, accuracy):
#     k = 0
#     kp1 = 0
#     kp2 = 0
#     kp3 = 0
#     kp4 = 0
#     kp5 = 0
#     km1 = 0
#     km2 = 0
#     km3 = 0
#     km4 = 0
#     km5 = 0
#     if derivative == 1 and accuracy == 2:
#         kp1 = 1/2
#         km1 = -1/2
#     else:
#         raise ValueError("koefficients for derivative:{} and accuracy:{} are not set"
#                          .format(derivative, accuracy))
#
#     return k, kp1, kp2, kp3, kp4, kp4, kp5, km1, km2, km3, km4, km5
#
# def central(i, func, step, der=1, acc=2):
#     k, kp1, kp2, kp3, kp4, kp4, kp5, km1, km2, km3, km4, km5 = \
#         central_koeff(der, acc)
#     central = k * func[i]
#     negative = km5*func[i - 5] + km4*func[i - 4] + km3*func[i - 3] + km2*func[i - 2] + km1*func[i - 1]
#     positive = kp5*func[i + 5] + kp4*func[i + 4] + kp3*func[i + 3] + kp2*func[i + 2] + kp1*func[i + 1]
#     return (negative + central +  positive) / (step ** int(der))
#
# def forward_koeff(derivative, accuracy):
#     k = 0
#     kp1 = 0
#     kp2 = 0
#     kp3 = 0
#     kp4 = 0
#     kp5 = 0
#     kp6 = 0
#     if derivative == 1 and accuracy == 1:
#         k = -1
#         kp1 = 1
#     elif derivative == 1 and accuracy == 2:
#         k = -3/2
#         kp1 = 2
#         kp2 = -1/2
#     else:
#         raise ValueError("koefficients for derivative:{} and accuracy:{} are not set"
#                          .format(derivative, accuracy))
#
#     return k, kp1, kp2, kp3, kp4, kp4, kp5, kp6
#
# def forward(i, func, step, der=1, acc=2):
#     k, kp1, kp2, kp3, kp4, kp4, kp5, kp6 = \
#         forward_koeff(der, acc)
#     positive = kp6*func[i + 6] + kp5*func[i + 5] + kp4*func[i + 4] + kp3*func[i + 3] + kp2*func[i + 2] + kp1*func[i + 1] +  k*func[i]
#     return positive / (step ** int(der))
#
# def backward_coeff(derivative, accuracy):
#     k = 0
#     km1 = 0
#     km2 = 0
#     km3 = 0
#     km4 = 0
#     km5 = 0
#     km6 = 0
#     if derivative == 1 and accuracy == 1:
#         k = 1
#         km1 = -1
#     elif derivative == 1 and accuracy == 2:
#         k = 3/2
#         km1 = -2
#         km2 = 1/2
#     else:
#         raise ValueError("koefficients for derivative:{} and accuracy:{} are not set"
#                          .format(derivative, accuracy))
#
#     return k, km1, km2, km3, km4, km4, km5, km6
#
# def backward(i, func, step, der=1, acc=2):
#     k, km1, km2, km3, km4, km4, km5, km6 = \
#         backward_coeff(der, acc)
#     negative = km6*func[i - 6] + km5*func[i - 5] + km4*func[i - 4] + km3*func[i - 3] + km2*func[i - 2] + km1*func[i - 1] +  k*func[i]
#     return negative / (step ** int(der))



#
def central2(i, func, step, der=1, acc=2):
    #
    if der == 1 and acc == 2:
        return ( (-1./2.)*func[i - 1] + 0*func[i] + (1./2.)*func[i + 1] ) / (step ** int(der))
    elif der == 1 and acc == 4:
        return ((1./12.) * func[i - 2] + (-2./3.) * func[i - 1] + 0 * func[i] + (2./3.) * func[i + 1] + (-1./12.) * func[i + 2]) / (step ** int(der))
    #
    elif der == 2 and acc == 2:
        return ((1.) * func[i - 1] + (-2.) * func[i] + (1.) * func[i + 1]) / (step ** int(der))
    elif der == 2 and acc == 4:
        return ((-1./12.) * func[i - 2] + (4./3.) * func[i - 1] + (-5./2.) * func[i] + (4./3.) * func[i + 1] + (-1./12.) * func[i + 2]) / (step ** int(der))
    else:
        raise ValueError("for central derivative:{} and accuracy:{} stencil is not".format(der, acc))

def forward2(i, func, step, der=1, acc=2):
    print(der, acc)
    if der == 1 and acc == 1:
        return (-1. * func[i] + 1. * func[i + 1]) / (step ** int(der))
    if der == 1 and acc == 2:
        return ( (-3./2.) * func[i] + 2. * func[i + 1] + (-1./2.) * func[i + 2]) / (step ** int(der))
    if der == 1 and acc == 3:
        return ( (-11./6.) * func[i] + (3.) * func[i + 1] + (-3./2.) * func[i + 2] + (1./3.) * func[i + 3]) / (step ** int(der))
    if der == 1 and acc == 4:
        return ( (-25./12.) * func[i] + (4.) * func[i + 1] + (-3.) * func[i + 2] + (4./3.) * func[i + 3] + (-1./4.) * func[i + 4]) / (step ** int(der))
    #
    if der == 2 and acc == 1:
        return ( (1.) * func[i] + (-2.) * func[i + 1] + (.1) * func[i + 2]) / (step ** int(der))
    if der == 2 and acc == 2:
        return ( (2.) * func[i] + (-5.) * func[i + 1] + (4.) * func[i + 2] + (-1.) * func[i + 3]) / (step ** int(der))
    if der == 2 and acc == 3:
        return ( (35./12.) * func[i] + (26./3.) * func[i + 1] + (19./2.) * func[i + 2] + (-14./3.) * func[i + 3] + (11./12.) * func[i + 4]) / (step ** int(der))
    if der == 2 and acc == 4:
        return ( (15./4.) * func[i] + (-77./6.) * func[i + 1] + (107./6.) * func[i + 2] + (-13.) * func[i + 3] + (61./12.) * func[i + 4] + (-5./6.) * func[i + 5]) / (step ** int(der))

    else:
        raise ValueError("for forward derivative:{} and accuracy:{} stencil is not".format(der, acc))

def backward2(i, func, step, der=1, acc=2):
    print(der, acc)
    if der == 1 and acc == 1:
        return (1. * func[i] + (-1.) * func[i + 1]) / (step ** int(der))
    if der == 1 and acc == 2:
        return ( (3./2.) * func[i] + (-2.) * func[i - 1] + (1./2.) * func[i - 2] ) / (step ** int(der))
    if der == 1 and acc == 3:
        return ( (11./6.) * func[i] + (-3.) * func[i - 1] + (3./2.) * func[i - 2] + (-1./3.) * func[i - 3]) / (step ** int(der))
    if int(der) == 1 and int(acc) == 4:
        # exit(1)
        return ( (25./12.) * func[i] + (-4.) * func[i - 1] + (3.) * func[i - 2] + (-4./3.) * func[i - 3] + (1./4.) * func[i - 4]) / (step ** int(der))
    #
    if der == 2 and acc == 1:
        return ( (-1.) * func[i] + (2.) * func[i - 1] + (-.1) * func[i - 2]) / (step ** int(der))
    if der == 2 and acc == 2:
        return ( (2.) * func[i] + (-5.) * func[i - 1] + (4.) * func[i - 2] + (-1.) * func[i - 3]) / (step ** int(der))
    if der == 2 and acc == 3:
        return ( (-35./12.) * func[i] + (-26./3.) * func[i - 1] + (-19./2.) * func[i - 2] + (14./3.) * func[i - 3] + (-11./12.) * func[i - 4]) / (step ** int(der))
    if der == 2 and acc == 4:
        return ( (15./4.) * func[i] + (-77./6.) * func[i - 1] + (107./6.) * func[i - 2] + (-13.) * func[i - 3] + (61./12.) * func[i - 4] + (-5./6.) * func[i - 5]) / (step ** int(der))
    else:
        raise ValueError("for backward derivative:{} and accuracy:{} stencil is not".format(der, acc))

def findif(f, n=41, x1=-1.00, x2=1.00, ord=1, acc=4, method="ghost"):

    if method == "onesided":
        print("using 2 one-sided for 2 accuracy")
        x = np.array(np.mgrid[x1:x2:n*1j], dtype=float)
        step = np.diff(x)[0]
        # print("grid: {}".format(x))
        # print("step: {}".format(step))
        #
        func = f(x)
        dfunc = np.zeros(n, dtype=np.float)
        #
        print("func: {}".format(func))
        # left boundary
        for i in range(0, acc): # [0, 1]
            # dfunc[i] = left_sided_second_order(i, func, step)
            dfunc[i] = forward2(i, func, step, der=ord, acc=acc)
        # right boundary
        for i in range(n - acc, n): #[n-2, n-1]
            # dfunc[i] = right_sided_second_order(i, func, step)
            dfunc[i] = backward2(i, func, step, der=ord, acc=acc)
        # central part
        if acc == 2:
            for i in range(1, n - 1): # [2, 3, ... n - 3]
                # dfunc[i] = central_second_order(i, func, step)
                dfunc[i] = central2(i, func, step, der=ord, acc=acc)
        elif acc == 4:
            for i in range(2, n - 2): # [2, 3, ... n - 3]
                # dfunc[i] = central_second_order(i, func, step)
                dfunc[i] = central2(i, func, step, der=ord, acc=acc)
        return dfunc
    elif method == "ghost":
        x = np.array(np.mgrid[x1:x2:n * 1j], dtype=float)
        # print(x)
        step = np.diff(x)[0]
        if acc == 2:
            print("using 2 ghosts for 2 accuracy")
            x = np.insert(x, 0, x[0] - step)
            x = np.insert(x, n + 1, x[-1] + step)
            func = f(x)
            dfunc = np.zeros(n + 2, dtype=np.float)

            for i in range(0, n + 1):  # [2, 3, ... n - 3]
                # dfunc[i] = central_second_order(i, func, step)
                dfunc[i] = central2(i, func, step, der=ord, acc=acc)
            res = dfunc[1:-1]
            assert len(res) == n
            return res
        if acc == 4:
            print("using 4 ghosts for 4 accuracy")
            x = np.insert(x, 0, x[0] - step)
            x = np.insert(x, 0, x[0] - step)
            x = np.insert(x, n+2, x[-1] + step)
            x = np.insert(x, n+3, x[-1] + step)
            print(x)
            # exit(1)
            func = f(x)
            dfunc = np.zeros(n+4, dtype=np.float)

            for i in range(0, n + 2): # [2, 3, ... n - 3]
                # dfunc[i] = central_second_order(i, func, step)
                dfunc[i] = central2(i, func, step, der=ord, acc=acc)
            res = dfunc[2:-2]
            assert len(res) == n
            return res

def convergence(f, df, ddf, ord, acc, method):
    #
    # ord = 2
    # acc = 4
    # method = "ghost"
    # method = "onesided"
    n = 41
    x1 = -0.5
    x2 = 0.5
    figpath = "/home/vsevolod/GIT/GitLab/projektpraktikum/WaveProject/fig/"
    #
    x1arr = np.array(np.mgrid[x1:x2:n * 1j], dtype=float)
    x2arr = np.array(np.mgrid[x1:x2:int(2 * n) * 1j], dtype=float)
    x4arr = np.array(np.mgrid[x1:x2:int(4 * n) * 1j], dtype=float)
    #
    if ord == 1:
        df_a1 = df(x1arr)
        df_a2 = df(x2arr)
        df_a4 = df(x4arr)
    elif ord == 2:
        df_a1 = ddf(x1arr)
        df_a2 = ddf(x2arr)
        df_a4 = ddf(x4arr)
    else:
        raise NameError("no analytical derivative for order: {}".format(ord))
    #
    df_n1 = findif(f, n=n, x1=x1, x2=x2, ord=ord, acc=acc, method=method)
    df_n2 = findif(f, n=int(2 * n), x1=x1, x2=x2, ord=ord, acc=acc, method=method)
    #
    diff1 = np.abs(df_a1 - df_n1) # same length
    diff2 = np.abs(df_a2 - df_n2) # same length
    #
    norm2 = np.sqrt(np.sum(diff1 ** 2)) / np.sqrt(np.sum(np.abs(diff2 ** 2)))
    print("Convergence: {} Expected(accuracy): {}".format(norm2, 2**float(acc)))

    ''' --------------------------------- '''

    fig = plt.figure()
    #
    ax = fig.add_subplot(211)
    if ord == 1:
        ax.plot(np.mgrid[x1:x2:10000j], df(np.mgrid[x1:x2:10000j]), ls='-', lw=0.5, color="blue", label=r"Anal")
    elif ord == 2:
        ax.plot(np.mgrid[x1:x2:10000j], ddf(np.mgrid[x1:x2:10000j]), ls='-', lw=0.5, color="blue", label=r"Anal")
    ax.plot(x1arr, df_n1, marker='o', linestyle = 'None', color="red", label=r"Num {}p".format(len(x1arr)))
    ax.plot(x2arr, df_n2, marker='x', linestyle = 'None', color="orange", label=r"Num {}p".format(len(x2arr)))
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.minorticks_on()
    ax.set_title("Order: {} Accuracy: {} Method: {} norm:{:.2f}".format(ord, acc, method, norm2))
    ax.legend(loc="lower left")
    ax.set_ylabel(r"$f(x)$", fontsize=14)
    ax.set_xticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    ax.set_xlabel('')
    ax.tick_params(labelbottom=False)
    #
    ax = fig.add_subplot(212)
    ax.plot(x1arr, diff1, marker='+', color="red", label=r"$|f_{E}' - f_{h}'|$") # linestyle = 'None',
    ax.plot(x2arr, diff2 * (2 ** acc), marker='x', color="orange", label=r"$|f_{E}' - f_{h/2}'|$") # linestyle = 'None',
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.set_yscale("log")
    ax.minorticks_on()
    # ax.set_title("Accuracy: {}".format(acc))
    ax.legend(loc="lower left")
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("norm", fontsize=14)
    #
    plt.subplots_adjust(hspace=0.0)
    plt.subplots_adjust(wspace=0.0)
    plt.tight_layout()
    plt.savefig(figpath + "{}_{}_{}_{}.png".format(str(f).split()[1], ord,acc,method), dpi=128)
    # plt.show()
    norm2 =  np.log2( np.sqrt(np.sum(diff1**2)) / np.sqrt(np.sum(np.abs(diff2**2))) )
    return norm2

def selfconvergence(f, ord, acc, method):
    #
    # ord = 2
    # acc = 4
    # method = "ghost"
    # method = "onesided"
    n = 41
    x1 = -0.5
    x2 = 0.5
    figpath = "/home/vsevolod/GIT/GitLab/projektpraktikum/WaveProject/fig/"
    #
    x1arr = np.array(np.mgrid[x1:x2:n * 1j], dtype=float)
    x2arr = np.array(np.mgrid[x1:x2:int(2*n) * 1j], dtype=float)
    x4arr = np.array(np.mgrid[x1:x2:int(4 * n) * 1j], dtype=float)
    #
    #
    df_n1 = findif(f, n=n, x1=x1, x2=x2, ord=ord, acc=acc, method=method)
    df_n2 = findif(f, n=int(2 * n), x1=x1, x2=x2, ord=ord, acc=acc, method=method)
    df_n4 = findif(f, n=int(4 * n), x1=x1, x2=x2, ord=ord, acc=acc, method=method)
    #
    # diff1 = np.abs(df_n1 - df_n2) # same length
    # diff2 = np.abs(df_n2 - df_n4) # same length
    #
    # norm2 = np.sqrt(np.sum(diff1 ** 2)) / np.sqrt(np.sum(np.abs(diff2 ** 2)))
    # print("Convergence: {} Expected(accuracy): {}".format(norm2, 2**float(acc)))

    ''' --------------------------------- '''

    fig = plt.figure()
    #
    ax = fig.add_subplot(111)
    ax.plot(x1arr, df_n1, marker='o', linestyle = 'None', color="red", label=r"Num {}p".format(len(x1arr)))
    ax.plot(x2arr, df_n2, marker='x', linestyle = 'None', color="orange", label=r"Num {}p".format(len(x2arr)))
    ax.plot(x4arr, df_n4, marker='+', linestyle='None', color="green", label=r"Num {}p".format(len(x4arr)))
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    ax.minorticks_on()
    ax.set_title("Order: {} Accuracy: {} Method: {} norm:{:.2f}".format(ord, acc, method, np.nan))
    ax.legend(loc="lower left")
    ax.set_ylabel(r"$f(x)$", fontsize=14)
    ax.set_xticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    ax.set_xlabel('')
    ax.tick_params(labelbottom=False)
    #
    # ax = fig.add_subplot(212)
    # ax.plot(x1arr, diff1, marker='+', color="red", label=r"$|f_{h}' - f_{h/2}'|$") # linestyle = 'None',
    # ax.plot(x4arr, diff2 * (2 ** acc), marker='x', color="orange", label=r"$|f_{h/2}' - f_{h/4}'|$") # linestyle = 'None',
    # ax.tick_params(
    #     axis='both', which='both', labelleft=True,
    #     labelright=False, tick1On=True, tick2On=True,
    #     labelsize=int(12),
    #     direction='in',
    #     bottom=True, top=True, left=True, right=True
    # )
    # ax.set_yscale("log")
    # ax.minorticks_on()
    # # ax.set_title("Accuracy: {}".format(acc))
    # ax.legend(loc="lower left")
    # ax.set_xlabel("$x$", fontsize=14)
    # ax.set_ylabel("norm", fontsize=14)
    # #
    plt.subplots_adjust(hspace=0.0)
    plt.subplots_adjust(wspace=0.0)
    plt.tight_layout()
    plt.savefig(figpath + "self_{}_{}_{}_{}.png".format(str(f).split()[1], ord,acc,method), dpi=128)
    # plt.show()
    # norm2 =  np.log2( np.sqrt(np.sum(diff1**2)) / np.sqrt(np.sum(np.abs(diff2**2))) )
    # return norm2

if __name__ == '__main__':

    for ord in [1,2]:
        for acc in [2, 4]:
            for method in ["ghost", "onesided"]:
                convergence(f1, df1, ddf1, ord, acc, method)
                convergence(f2, df2, ddf2, ord, acc, method)
                convergence(f3, df3, ddf3, ord, acc, method)
                convergence(f4, df4, ddf4, ord, acc, method)
                convergence(f5, df5, ddf5, ord, acc, method)
                convergence(f6, df6, ddf6, ord, acc, method)
                selfconvergence(f1, ord, acc, method)
                selfconvergence(f2, ord, acc, method)
                selfconvergence(f3, ord, acc, method)
                selfconvergence(f4, ord, acc, method)
                selfconvergence(f5, ord, acc, method)
                selfconvergence(f6, ord, acc, method)
    #

    # df2nd(f1)
    # #
    # n = 40  # grid points

    # x = np.zeros(n + 1, dtype=np.float)  # array to store values of x
    # step = 0.02 / float(n)  # step size
    #
    # f = np.zeros(n + 1, dtype=np.float)  # array to store values of f
    # df = np.zeros(n + 1, dtype=np.float)  # array to store values of calulated derivative
    #
    # for i in range(0, n + 1):  # adds values to arrays for x and f(x)
    #     x[i] = -0.01 + np.float(i) * step
    #     f[i] = f1(x[i])
    #
    # # print("grid: {}".format(x))
    # # print("step: {}".format(step))
    # # exit(1)
    # # have to calculate end points seperately using one sided form
    #
    # # first order lieft side
    # # df[0] = (f[1] - f[0]) / step
    # # df[1] = (f[2] - f[1]) / step
    #
    # # first order right side
    # # df[n-1] = (f[n - 1] - f[n - 2]) / step
    # # df[n] = (f[n] - f[n-1]) / step
    #
    # # second order left side
    # i = 0
    # df[i] = (-3 * f[i] + 4 * f[i + 1] - f[i+2]) /(2*step)
    # i = 1
    # df[i] = (-3 * f[i] + 4 * f[i + 1] - f[i+2]) /(2*step)
    #
    # # second order right side
    # i = n-1
    # df[i] = (3 * f[i] - 4 * f[i-1] + f[i-2]) / (2*step)
    # i = n
    # df[i] = (3 * f[i] - 4 * f[i-1] + f[i-2]) / (2*step)

    # second order
    # df[0] = (f[2] - 2 * f[1] + f[0]) / step ** 2
    # df[1] = (f[3] - 2 * f[2] + f[1]) / step ** 2
    # df[n - 1] = (f[n - 1] - 2 * f[n - 2] + f[n - 3]) / step ** 2
    # df[n] = (f[n] - 2 * f[n - 1] + f[n - 2]) / step ** 2
    #
    # for i in range(2, n - 1):  # add values to array for derivative
    #     df[i] = (f[i + 1] - f[i - 1]) / (2 * step)  # first derivative
        # df[i] = (f[i + 1] - 2 * f[i] + f[i - 1]) / step ** 2 # second derivative
    #
    # n = 1001
    # x1 = -0.5
    # x2 = 0.5
    # x = np.array(np.mgrid[x1:x2:n * 1j], dtype=float)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ax.plot(x, df1(x), marker='o', color='red', label="anal 1st order")
    # # ax.plot(x, df2nd(f1,n=n,x1=x1,x2=x2, ord=1, acc=4), marker='x', color = 'blue', alpha=0.6, label="num 1st order")
    # #
    # ax.plot(x, ddf1(x), marker='d', color='red', label='anal 2nd order')
    # ax.plot(x, df2nd(f1,n=n,x1=x1,x2=x2, ord=2, acc=4), marker='s', color = 'blue', alpha=0.6, label='num 2nd order')
    # #
    # plt.legend(loc="best")
    #
    # #
    # plt.tight_layout()
    # plt.show()

print("hello world")
