"""
 Implementation of the finite differencing
 function 'finite_differencing(f, n, x1, x2, ord, acc, method)', that performs the finite differencing of a function f(x),
 using $n$ equally spaced points between $x1$ and $x2$, using the stencil of order $ord$ and accuracy $acc$ with method
 method. \\
 Possible orders: $[1,2]$ for the first and second order derivative \\
 Possible accuracies: $[2,4]$ for the second and 4th order accuracy stencils \\
 Possible methods: ghosts, one-sided. The first one will extend the $x$ to 1 (2) points in left and right for second
 (forth) accuracies and computes the central finite differencing for points between $0$ to $n$, using added points as
 ghosts. Second method uses one-sided stencils for 1 (2) border points, where central stencil cannot be used. \\
 To write stencils the coefficients from  were used
"""
# \url{https://en.wikipedia.org/wiki/Finite_difference_coefficient}

# TODO Convergence order is not reproduced!

import numpy as np
import matplotlib.pyplot as plt

FIGPATH = "../figs/finite_differencing/"

class Functions:
    @staticmethod
    def f1(x):
        return (x - 0.5) ** 2. + x

    @staticmethod
    def df1(x):
        return 2. * x

    @staticmethod
    def ddf1(x):
        res = np.zeros(len(x))
        res.fill(2.)
        return res

    @staticmethod
    def f2(x):
        return (x - 0.5) ** 3. + (x - 0.5) ** 2. + x

    @staticmethod
    def df2(x):
        return 3. * x ** 2. - x + (3. / 4.)

    @staticmethod
    def ddf2(x):
        return 6. * x - 1.

    @staticmethod
    def f3(x):
        return np.sqrt(x)

    @staticmethod
    def df3(x):
        return 1. / (2. * np.sqrt(x))

    @staticmethod
    def ddf3(x):
        return -1. / (4. * x ** (3. / 2.))

    @staticmethod
    def f4(x):
        return np.sin(12. * np.pi * x)

    @staticmethod
    def df4(x):
        return 12 * np.pi * np.cos(12. * np.pi * x)

    @staticmethod
    def ddf4(x):
        return -144. * np.pi ** 2 * np.sin(12. * np.pi * x)

    @staticmethod
    def f5(x):
        return np.sin(12. * np.pi * x) ** 4.

    @staticmethod
    def df5(x):
        return 48. * np.pi * np.cos(12. * np.pi * x) * np.sin(12. * np.pi * x) ** 3.

    @staticmethod
    def ddf5(x):
        return -576. * np.pi ** 2. * np.sin(12. * np.pi * x) ** 2. * (
                    -3 * np.cos(12 * np.pi * x) ** 2. + np.sin(12 * np.pi * x) ** 2.)

    @staticmethod
    def f6(x, a=5.):
        return np.exp(-a * x ** 2.)

    @staticmethod
    def df6(x, a=5.):
        return -2. * a * np.exp(-a * x ** 2.) * x

    @staticmethod
    def ddf6(x, a=5.):
        return 2. * a * np.exp(- a * x ** 2.) * (-1. + 2. * a * x ** 2.)


class Stenciels:

    @staticmethod
    def central2(i, func, step, der=1, acc=2):
        #
        if der == 1 and acc == 2:
            return ((-1. / 2.) * func[i - 1] + 0 * func[i] + (1. / 2.) * func[i + 1]) / (step ** int(der))
        elif der == 1 and acc == 4:
            return ((1. / 12.) * func[i - 2] + (-2. / 3.) * func[i - 1] + 0 * func[i] + (2. / 3.) * func[i + 1] + (
                        -1. / 12.) * func[i + 2]) / (step ** int(der))
        #
        elif der == 2 and acc == 2:
            return ((1.) * func[i - 1] + (-2.) * func[i] + (1.) * func[i + 1]) / (step ** int(der))
        elif der == 2 and acc == 4:
            return ((-1. / 12.) * func[i - 2] + (4. / 3.) * func[i - 1] + (-5. / 2.) * func[i] + (4. / 3.) * func[
                i + 1] + (-1. / 12.) * func[i + 2]) / (step ** int(der))
        else:
            raise ValueError("for central derivative:{} and accuracy:{} stencil is not".format(der, acc))

    @staticmethod
    def forward2(i, func, step, der=1, acc=2):
        # print(der, acc)
        if der == 1 and acc == 1:
            return (-1. * func[i] + 1. * func[i + 1]) / (step ** int(der))
        if der == 1 and acc == 2:
            return ((-3. / 2.) * func[i] + 2. * func[i + 1] + (-1. / 2.) * func[i + 2]) / (step ** int(der))
        if der == 1 and acc == 3:
            return ((-11. / 6.) * func[i] + (3.) * func[i + 1] + (-3. / 2.) * func[i + 2] + (1. / 3.) * func[i + 3]) / (
                        step ** int(der))
        if der == 1 and acc == 4:
            return ((-25. / 12.) * func[i] + (4.) * func[i + 1] + (-3.) * func[i + 2] + (4. / 3.) * func[i + 3] + (
                        -1. / 4.) * func[i + 4]) / (step ** int(der))
        #
        if der == 2 and acc == 1:
            return ((1.) * func[i] + (-2.) * func[i + 1] + (.1) * func[i + 2]) / (step ** int(der))
        if der == 2 and acc == 2:
            return ((2.) * func[i] + (-5.) * func[i + 1] + (4.) * func[i + 2] + (-1.) * func[i + 3]) / (
                        step ** int(der))
        if der == 2 and acc == 3:
            return ((35. / 12.) * func[i] + (26. / 3.) * func[i + 1] + (19. / 2.) * func[i + 2] + (-14. / 3.) * func[
                i + 3] + (11. / 12.) * func[i + 4]) / (step ** int(der))
        if der == 2 and acc == 4:
            return ((15. / 4.) * func[i] + (-77. / 6.) * func[i + 1] + (107. / 6.) * func[i + 2] + (-13.) * func[
                i + 3] + (61. / 12.) * func[i + 4] + (-5. / 6.) * func[i + 5]) / (step ** int(der))

        else:
            raise ValueError("for forward derivative:{} and accuracy:{} stencil is not".format(der, acc))

    @staticmethod
    def backward2(i, func, step, der=1, acc=2):
        # print(der, acc)
        if der == 1 and acc == 1:
            return (1. * func[i] + (-1.) * func[i + 1]) / (step ** int(der))
        if der == 1 and acc == 2:
            return ((3. / 2.) * func[i] + (-2.) * func[i - 1] + (1. / 2.) * func[i - 2]) / (step ** int(der))
        if der == 1 and acc == 3:
            return ((11. / 6.) * func[i] + (-3.) * func[i - 1] + (3. / 2.) * func[i - 2] + (-1. / 3.) * func[i - 3]) / (
                        step ** int(der))
        if int(der) == 1 and int(acc) == 4:
            # exit(1)
            return ((25. / 12.) * func[i] + (-4.) * func[i - 1] + (3.) * func[i - 2] + (-4. / 3.) * func[i - 3] + (
                        1. / 4.) * func[i - 4]) / (step ** int(der))
        #
        if der == 2 and acc == 1:
            return ((-1.) * func[i] + (2.) * func[i - 1] + (-.1) * func[i - 2]) / (step ** int(der))
        if der == 2 and acc == 2:
            return ((2.) * func[i] + (-5.) * func[i - 1] + (4.) * func[i - 2] + (-1.) * func[i - 3]) / (
                        step ** int(der))
        if der == 2 and acc == 3:
            return ((-35. / 12.) * func[i] + (-26. / 3.) * func[i - 1] + (-19. / 2.) * func[i - 2] + (14. / 3.) * func[
                i - 3] + (-11. / 12.) * func[i - 4]) / (step ** int(der))
        if der == 2 and acc == 4:
            return ((15. / 4.) * func[i] + (-77. / 6.) * func[i - 1] + (107. / 6.) * func[i - 2] + (-13.) * func[
                i - 3] + (61. / 12.) * func[i - 4] + (-5. / 6.) * func[i - 5]) / (step ** int(der))
        else:
            raise ValueError("for backward derivative:{} and accuracy:{} stencil is not".format(der, acc))


def finite_differencing(f, n=41, x1=-1.00, x2=1.00, ord=1, acc=4, method="ghost"):

    if method == "onesided":
        # print("using 2 one-sided for 2 accuracy")
        x = np.array(np.mgrid[x1:x2:n*1j], dtype=float)
        step = np.diff(x)[0]
        # print("grid: {}".format(x))
        # print("step: {}".format(step))
        #
        func = f(x)
        dfunc = np.zeros(n, dtype=np.float)
        #
        # print("func: {}".format(func))
        # left boundary
        for i in range(0, acc): # [0, 1]
            # dfunc[i] = left_sided_second_order(i, func, step)
            dfunc[i] = Stenciels.forward2(i, func, step, der=ord, acc=acc)
        # right boundary
        for i in range(n - acc, n): #[n-2, n-1]
            # dfunc[i] = right_sided_second_order(i, func, step)
            dfunc[i] = Stenciels.backward2(i, func, step, der=ord, acc=acc)
        # central part
        if acc == 2:
            for i in range(1, n - 1): # [2, 3, ... n - 3]
                # dfunc[i] = central_second_order(i, func, step)
                dfunc[i] = Stenciels.central2(i, func, step, der=ord, acc=acc)
        elif acc == 4:
            for i in range(2, n - 2): # [2, 3, ... n - 3]
                # dfunc[i] = central_second_order(i, func, step)
                dfunc[i] = Stenciels.central2(i, func, step, der=ord, acc=acc)
        return dfunc
    elif method == "ghost":
        x = np.array(np.mgrid[x1:x2:n * 1j], dtype=float)
        # print(x)
        step = np.diff(x)[0]
        if acc == 2:
            # print("using 2 ghosts for 2 accuracy")
            x = np.insert(x, 0, x[0] - step)
            x = np.insert(x, n + 1, x[-1] + step)
            func = f(x)
            dfunc = np.zeros(n + 2, dtype=np.float)

            for i in range(0, n + 1):  # [2, 3, ... n - 3]
                # dfunc[i] = central_second_order(i, func, step)
                dfunc[i] = Stenciels.central2(i, func, step, der=ord, acc=acc)
            res = dfunc[1:-1]
            assert len(res) == n
            return res
        if acc == 4:
            # print("using 4 ghosts for 4 accuracy")
            x = np.insert(x, 0, x[0] - step)
            x = np.insert(x, 0, x[0] - step)
            x = np.insert(x, n+2, x[-1] + step)
            x = np.insert(x, n+3, x[-1] + step)
            # print(x)
            # exit(1)
            func = f(x)
            dfunc = np.zeros(n+4, dtype=np.float)

            for i in range(0, n + 2): # [2, 3, ... n - 3]
                # dfunc[i] = central_second_order(i, func, step)
                dfunc[i] = Stenciels.central2(i, func, step, der=ord, acc=acc)
            res = dfunc[2:-2]
            assert len(res) == n
            return res


def convergence_finite_diff(f, df, ddf, ord, acc, method, n=41, x1=-0.5, x2=0.5, plot=True):

    # numenator
    x1arr = np.array(np.mgrid[x1:x2:n * 1j], dtype=float)
    # --- analytical derivative
    if ord == 1: df_a1 = df(x1arr)
    elif ord == 2: df_a1 = ddf(x1arr)
    else: raise NameError("no analytical derivative for order: {}".format(ord))
    # --- numerical derivative
    df_n1 = finite_differencing(f, n=n, x1=x1, x2=x2, ord=ord, acc=acc, method=method)

    diff1 = np.abs(df_a1 - df_n1)  # same length

    # denumenator
    x2arr = np.array(np.mgrid[x1:x2:int(2 * n) * 1j], dtype=float)
    # --- analytical
    if ord == 1: df_a2 = df(x2arr)
    elif ord == 2: df_a2 = ddf(x2arr)
    else: raise NameError("no analytical derivative for order: {}".format(ord))
    # --- numerical
    df_n2 = finite_differencing(f, n=int(2 * n), x1=x1, x2=x2, ord=ord, acc=acc, method=method)

    diff2 = np.abs(df_a2 - df_n2)  # same length

    # final
    norm2 = np.sqrt(np.sum(diff1 ** 2)) / np.sqrt(np.sum(np.abs(diff2 ** 2)))
    print("Convergence: {} Expected(accuracy): {}".format(norm2, 2 ** float(acc)))

    if plot:
        fig = plt.figure()
        #
        ax = fig.add_subplot(211)
        if ord == 1:
            ax.plot(np.mgrid[x1:x2:10000j], df(np.mgrid[x1:x2:10000j]), ls='-', lw=0.5, color="blue", label=r"Anal")
        elif ord == 2:
            ax.plot(np.mgrid[x1:x2:10000j], ddf(np.mgrid[x1:x2:10000j]), ls='-', lw=0.5, color="blue", label=r"Anal")
        ax.plot(x1arr, df_n1, marker='o', linestyle='None', color="red", label=r"Num {}p".format(len(x1arr)))
        ax.plot(x2arr, df_n2, marker='x', linestyle='None', color="orange", label=r"Num {}p".format(len(x2arr)))
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
        ax.plot(x1arr, diff1, marker='+', color="red", label=r"$|f_{E}' - f_{h}'|$")  # linestyle = 'None',
        ax.plot(x2arr, diff2 * (2 ** acc), marker='x', color="orange",
                label=r"$|f_{E}' - f_{h/2}'|$")  # linestyle = 'None',
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
        plt.savefig(FIGPATH + "{}_{}_{}_{}.png".format(str(f).split()[1], ord, acc, method), dpi=128)
        # plt.show()
        norm2 = np.log2(np.sqrt(np.sum(diff1 ** 2)) / np.sqrt(np.sum(np.abs(diff2 ** 2))))
        return norm2


def selfconvergence_finite_diff(f, ord, acc, method, n=41, x1=-0.5, x2=0.5, plot=True):
    #
    # ord = 2
    # acc = 4
    # method = "ghost"
    # method = "onesided"
    #
    x1arr = np.array(np.mgrid[x1:x2:n * 1j], dtype=float)
    x2arr = np.array(np.mgrid[x1:x2:int(2*n) * 1j], dtype=float)
    x4arr = np.array(np.mgrid[x1:x2:int(4 * n) * 1j], dtype=float)
    #
    #
    df_n1 = finite_differencing(f, n=n, x1=x1, x2=x2, ord=ord, acc=acc, method=method)
    df_n2 = finite_differencing(f, n=int(2 * n), x1=x1, x2=x2, ord=ord, acc=acc, method=method)
    df_n4 = finite_differencing(f, n=int(4 * n), x1=x1, x2=x2, ord=ord, acc=acc, method=method)
    #
    diff1 = np.abs(df_n1 - df_n2[::2]) # same length
    diff2 = np.abs(df_n2[::2] - df_n4[::4]) # same length

    norm2 = np.sqrt(np.sum(diff1 ** 2)) / np.sqrt(np.sum(np.abs(diff2 ** 2)))
    print("Self-Convergence: {} Expected(accuracy): {}".format(norm2, 2**float(acc)))

    ''' --------------------------------- '''
    if plot:
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
        plt.savefig(FIGPATH + "self_{}_{}_{}_{}.png".format(str(f).split()[1], ord,acc,method), dpi=128)
    # plt.show()
    # norm2 =  np.log2( np.sqrt(np.sum(diff1**2)) / np.sqrt(np.sum(np.abs(diff2**2))) )
    # return norm2


def run_all_convergnece():
    for ord in [1,2]:
        for acc in [2, 4]:
            for method in ["ghost", "onesided"]:
                convergence_finite_diff(Functions.f1, Functions.df1, Functions.ddf1, ord, acc, method)
                convergence_finite_diff(Functions.f2, Functions.df2, Functions.ddf2, ord, acc, method)
                convergence_finite_diff(Functions.f3, Functions.df3, Functions.ddf3, ord, acc, method)
                convergence_finite_diff(Functions.f4, Functions.df4, Functions.ddf4, ord, acc, method)
                convergence_finite_diff(Functions.f5, Functions.df5, Functions.ddf5, ord, acc, method)
                convergence_finite_diff(Functions.f6, Functions.df6, Functions.ddf6, ord, acc, method)

def run_all_self_convergence():
    for ord in [1,2]:
        for acc in [2, 4]:
            for method in ["ghost", "onesided"]:
                selfconvergence_finite_diff(Functions.f1, ord, acc, method)
                selfconvergence_finite_diff(Functions.f2, ord, acc, method)
                selfconvergence_finite_diff(Functions.f3, ord, acc, method)
                selfconvergence_finite_diff(Functions.f4, ord, acc, method)
                selfconvergence_finite_diff(Functions.f5, ord, acc, method)
                selfconvergence_finite_diff(Functions.f6, ord, acc, method)

if __name__ == '__main__':

    run_all_convergnece()

    run_all_self_convergence()
