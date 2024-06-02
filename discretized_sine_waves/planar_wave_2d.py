import numpy as np
from matplotlib import cm
import pylab
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from brainblocks.tools import HyperGridTransform


def makeData(x_bin_size=1.0, y_bin_size=1.0, x_period=16.0, y_period=16.0, step_size=0.05):

    # mesh values over the plane
    x = np.arange(-20, 20, step_size)
    y = np.arange(-20, 20, step_size)
    xgrid, ygrid = np.meshgrid(x, y)

    # values binned into regular intervals, return the minimum of that interval
    xgrid_bin = discretize_by_bin_size(xgrid, x_bin_size)
    ygrid_bin = discretize_by_bin_size(ygrid, y_bin_size)

    #xgrid_mod = np.fmod(xgrid, x_period)
    #ygrid_mod = np.fmod(ygrid, y_period)
    #zgrid = np.sin(xgrid_bin*np.pi/4.) * np.sin(ygrid_bin*np.pi/4.)

    print(xgrid_bin[0,:])

    # xgrid_bin (mod x_period)
    #zgrid = np.mod(xgrid_bin, x_period)
    zgrid=xgrid_bin

    print(zgrid[0,:])

    # wave function values in z-axis

    # composite sine wave
    #zgrid = np.cos(xgrid_bin/x_period*2.*np.pi) * np.sin(ygrid_bin/y_period*2.*np.pi)
    
    # composite sine wave
    #zgrid = np.sin(xgrid_bin/x_period*2.*np.pi) * np.cos(ygrid_bin/y_period*2.*np.pi)

    # circular ripple sine wave
    #d = np.sqrt((xgrid_bin/x_period*2.*np.pi)**2 + (ygrid_bin/y_period*2.*np.pi)**2)
    #zgrid = np.sin(d)
    
    # values for plotting 3D function
    return xgrid, ygrid, zgrid


def discretize_by_bin_size(x, bin_size):
    # values binned into regular intervals, return the minimum of that interval

    # scale values so that integers are size of bin
    scale = 1 / bin_size
    x_scaled = scale * x

    # minimum value of each interval using floor function
    x_scaled_bin = np.floor(x_scaled)

    # rescale values to get non-integer floors
    x_bin = (1/scale) * x_scaled_bin

    return x_bin



#x1, y1, z1 = makeData(1.0, 1.0, 16.0, 16.0)
#x2, y2, z2 = makeData(0.5, 0.5, 8.0, 8.0)
#x3, y3, z3 = makeData(0.25, 0.25, 4.0, 4.0)

x, y, z = makeData(1.0, 1.0, 8.0, 8.0)

#x1, y1, z1 = makeData(1.0, 1.0, 16.0, 16.0)
#x2, y2, z2 = makeData(0.5, 0.5, 8.0, 8.0)
#x = x1+x2
#y = y1+y2
#z = z1+z2

#x = x1+x2+x3
#y = y1+y2+y3
#z = z1+z2+z3

fig = pylab.figure()
axes = fig.add_subplot()

#colors = sns.color_palette("deep", n_colors=num_grids)

#contour_set = axes.contourf(x, y, z, levels=10, norm=no_norm, cmap=gray_cmap)
#contour_set = axes.contourf(x, y, z, cmap=cm.Set1)
contour_set = axes.contourf(x, y, z, levels=8, cmap=cm.Dark2)

#pylab.show()

#axes = Axes3D(fig)
#axes.plot_surface(x, y, z, cmap=cm.jet)
#axes.plot_surface(x, y, z, cmap=cm.binary)
#axes.plot_surface(x, y, z, cmap=cm.gist_rainbow)
#axes.plot_surface(x, y, z, cmap=cm.coolwarm)
#axes.plot_surface(x, y, z, cmap=cm.seismic)
#axes.plot_surface(x, y, z, cmap=cm.Set1) #tab20
#axes.set_zlim(0,3)
#axes.set_zlim(-3,3)
pylab.savefig("image2.png")

