import glob
import os
import seaborn as sns

# brainblocks
from brainblocks.tools import HyperGridTransform

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, NoNorm
from matplotlib import ticker

import numpy as np


def run(filename_root="visual_test_frame_1", output_dir="out"):
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create filename path
    filename_str = output_dir + "/" + filename_root + "_%05u.png"

    # remove old files if exist
    results = glob.glob(output_dir + "/" + filename_root + "_*")
    for filename in results:
        os.remove(filename)

    # fig = plt.figure(num=1)
    fig = plt.figure(num=1, figsize=(8, 8))  # , constrained_layout=True)
    fig, axes = plt.subplots(1, 1, num=1)  # , gridspec_kw={"width_ratios": [0.6, 0.5]})

    # axes for hypergrid visual
    # ax0 = axes[0]
    ax0 = axes

    fig.subplots_adjust(left=0.3, right=0.8, bottom=0.3, top=0.7)  # , wspace=0.2)

    # points for initializing HyperGrid Transform
    points = np.array([[0.0, ]])

    num_grids = 1
    num_bins = 8
    num_subspace_dims = 1
    step_size = 0.05

    # grid for visualizing
    hgt = HyperGridTransform(num_grids=num_grids, num_bins=num_bins, num_subspace_dims=num_subspace_dims,
                             use_standard_bases=True).fit(points)

    # converted feature points to gnomes
    bits_tensor = hgt.transform(points)
    print(bits_tensor)

    # color palette to distinguish the grids
    sns.set()
    colors = sns.color_palette("deep", n_colors=hgt.num_bins)

    # colors for the gnome heatmap representation converted to RGBA, prepending white
    colors2 = [np.array(colorConverter.to_rgba("white"))]
    for k in range(hgt.num_grids):
        curr_color = np.array(colorConverter.to_rgba(colors[k]))
        colors2.append(curr_color)


    # scaled from unit vectors to original period-sized norms
    scaled_bases = np.array([np.multiply(hgt.subspace_vectors[k], hgt.subspace_periods[k].reshape(-1, 1)) for k in
                             range(hgt.subspace_vectors.shape[0])])

    ax0.set_aspect('equal')
    ax0.set_xlim(-1.4, 1.4)
    ax0.set_ylim(-1.4, 1.4)
    ax0.tick_params(labelsize=8)
    tick_locs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    ax0.xaxis.set_major_locator(ticker.FixedLocator(tick_locs))
    ax0.yaxis.set_major_locator(ticker.FixedLocator(tick_locs))

    # trajectory for moving point input
    xs = np.arange(-1, 1, step_size)
    ys = np.zeros_like(xs)
    num_points = len(xs)

    lines = None
    paths = None
    contour_set = None

    x = np.arange(-1.4, 1.4, step_size)
    y = np.arange(-1.4, 1.4, step_size)
    xgrid, ygrid = np.meshgrid(x, y)

    print(xgrid.shape)
    # X : array of shape (num_samples, num_features)
    x_flat = xgrid.reshape(-1, 1)
    print(x_flat.shape)
    x_transform = hgt.transform(x_flat).astype(int)

    print(x_transform.shape)
    indices = np.where(x_transform == 1)
    x_transform[indices] = indices[2] + 1
    print(x_transform.shape)



    xgrid_transform = x_transform.reshape(xgrid.shape[0], xgrid.shape[1], num_bins).astype(int)
    print(xgrid_transform.shape)
    #print(xgrid_transform)
    zgrid = xgrid

    for point_index in range(num_points):

        points = [(xs[point_index],)]

        if not paths is None:
            paths.remove()

        paths = ax0.scatter(xs[point_index], ys[point_index], edgecolor='k', color='w', marker='o', s=60, zorder=100)

        point = [(xs[point_index],)]

        # assume one point, remove excess axis for single point, convert to int
        X_gnome = hgt.transform(point)
        X_gnome = X_gnome.reshape(X_gnome.shape[1:]).astype(int)




        # purge previous contours, generate new contours
        if not contour_set is None:
            for coll in contour_set.collections:
                coll.remove()
        contour_set = ax0.contourf(xgrid, ygrid, zgrid, levels=num_bins, colors=colors2)


        #contour_set = draw_similarity(ax0, hgt, X_gnomes)

        # give positive integer label for each grid's active bits, where label is the grid number
        #indices = np.where(X_gnomes == 1)
        #X_gnomes[indices] = indices[0] + 1

        #draw_gnomes(ax0, X_gnomes, hgt.num_bins, num_grids=hgt.num_grids, colors=colors2)

        print("saving %s" % (filename_str % point_index))
        plt.savefig(filename_str % point_index, bbox_inches='tight')

    plt.clf()


# plot fading line

if __name__ == "__main__":
    run()
