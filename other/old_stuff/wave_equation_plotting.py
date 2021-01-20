# -*- coding: utf-8 -*-
"""
Tests and Plotting for the Projektparktikum on Wave Equation

For Animations use >>%matplotlib qt<< in console or Tools>Preferences>IPython console>Graphics>Backend: Qt

"""
import numpy as np
import matplotlib.pyplot as plt
from wave_equation_functionsb import *
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# set input

alpha = 0.5  # Currant Number should be less than 1
boundaries = 'o3'

# initialize grid
Nx = 800
xstart = -5
xend = 5
dx = (xend - xstart) / (Nx)
# avoid zeros Avoid common multiples with transcendent factor 1/pi
x = np.linspace(xstart + dx / np.pi, xend + dx / np.pi, Nx + 1)

periods = 2

# create array with initial values, i.e. choose a function
phi0 = gaussian(x, 0, 2)
# phi0 = sine(x,2)
# phi0  = pulse(x,1,10)
pi0 = np.zeros_like(x)

# call function
output, x, t = waveeq(phi0, pi0, x, periods, alpha, boundaries, outputstep=1)

phi = output[:, 0, :]
pi = output[:, 1, :]

'''
Plotting functions

1:  heatmap of field phi over x and t 

2:  snapshot of phi and Pi at timestep 'plotstep'

3:  animation

'''

########################
# 1 plot at time slice #
########################

# plotstep=10                                             #direct timestep input
plotstep = round(0.23 * len(output[0, 0, :]))  # percentage of total time evolution
phimax = np.max(abs(phi[:, plotstep]))
pimax = np.max(abs(pi[:, plotstep]))
allmax = np.max([abs(phi[:, plotstep]), abs(pi[:, plotstep])])

fig2, ax2 = plt.subplots(figsize=(12, 8))

ax2.set_title('$\phi$ and $\Pi$ at timestep %i' % (plotstep))
ax2.set_xlabel('x')
ax2.set_ylabel('field amplitude $\phi$ [a.u.]', color='r')
ax2.set_xlim([xstart, xend])
ax2.set_ylim([-1.1 * allmax, 1.1 * allmax])
ax2.grid(True)
l1 = ax2.plot(x, phi[:, plotstep], 'r-', label='$\phi$')

ax3 = ax2.twinx()

ax3.set_ylabel('field amplitude $\Pi$ [a.u.]', color='b')
# ax3.set_ylim([-1.1*pimax,1.1*pimax])
ax3.set_ylim([-1.1 * allmax, 1.1 * allmax])
# ax3.set_yticks(np.linspace(ax3.get_yticks()[0], ax3.get_yticks()[-1], len(ax2.get_yticks()))) # put -2 no of steps if grids are not aligned
ax3.grid(False)

l2 = ax3.plot(x, pi[:, plotstep], 'b-', label='$\Pi$')

# complicated legend due to double axis
lns = l1 + l2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc='upper left')

fig2.tight_layout()

# plt.savefig('timeslices.png', dpi=300)
plt.show()

#############
# 2 heatmap #
#############

fig, ax = plt.subplots()
im = ax.pcolormesh(x, t, output[:, 0, 1:].T, vmin=-phimax, vmax=phimax, cmap='RdBu')
ax.set_title('field amplitude $\phi$ [a.u.]')
ax.set_xlabel('x [a.u.]')
ax.set_ylabel('$n_t$ [a.u.]')
fig.colorbar(im, ax=ax)
plt.show()

###############
# 3 animation #
###############
# uses steps from https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/

plt.style.use('seaborn-pastel')

fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(-5.5, 5.5), ylim=(-1.5, 1.5))
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False)

line1 = ax.plot(x, phi[:, 0], color='r', lw=2)[0]
line2 = ax.plot(x, pi[:, 0], color='b', lw=2)[0]


def animate(i):
    line1.set_ydata(phi[:, i])
    line2.set_ydata(pi[:, i])


anim = FuncAnimation(fig, animate, interval=10, frames=len(t), repeat_delay=500)
# anim.save('anime.gif')
plt.draw()
plt.show()

'''
################
# 4 seismogram #
################

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

skip = 30
X = t[::int(Nt/skip)+1]
Y = x
X,Y = np.meshgrid(X,Y)

Z = np.zeros((Nx,skip))

for i in range(skip):
    Z[:,i] = phi[:,i*int(Nt/skip)]
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolor='w', color='w', shade=False, lw=.5)

#ax.set_zlim(0, 5)
#ax.set_xlim(-51, 51)
ax.set_zlabel("Intensity")
ax.view_init(30,200)                # hight, rotation in deg
plt.show()
'''