import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


def midpoints(x):
    print("midpoints", x.ndim)
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        #print(x)
        sl += np.index_exp[:]
        print(sl)
        print(np.index_exp[:], np.index_exp[:-1], np.index_exp[1:])
    return x

# prepare some coordinates, and attach rgb values to each
r, theta, z = np.mgrid[0:1:11j, 0:np.pi*2:8j, -0.5:0.5:11j]
x = r*np.cos(theta)
y = r*np.sin(theta)

print(r)
rc, thetac, zc = midpoints(r), midpoints(theta), midpoints(z)

print(rc)
#print(thetac)
#print(zc)

# define a wobbly torus about [0.7, *, 0]
annulus_radius = (r < 3) & (y < 3) & (z < 3)
annulus_z = (x < 3) & (y < 3) & (z < 3)
annulus = (rc >= 0.7) & (zc >= 0.0) & (zc < 0.1) 

sphere = annulus

#sphere = (rc - 0.7)**2 + (zc + 0.2*np.cos(thetac*2))**2 < 0.2**2

#cube1 = (x < 3) & (y < 3) & (z < 3)
#cube2 = (x >= 5) & (y >= 5) & (z >= 5)

# combine the color components
hsv = np.zeros(sphere.shape + (3,))
hsv[..., 0] = thetac / (np.pi*2)
hsv[..., 1] = rc
hsv[..., 2] = zc + 0.5
colors = matplotlib.colors.hsv_to_rgb(hsv)

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x, y, z, sphere,
          facecolors=colors,
          edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
          linewidth=0.5)

plt.show()



"""
# prepare some coordinates
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between
# them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# combine the objects into a single boolean array
voxelarray = cube1 | cube2 | link

# set the colors of each object
colors = np.empty(voxelarray.shape, dtype=object)
colors[link] = 'red'
colors[cube1] = 'blue'
colors[cube2] = 'green'

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
"""


