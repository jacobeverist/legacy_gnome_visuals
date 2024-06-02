import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import colorConverter
from matplotlib import ticker
import matplotlib.patches as patches
import random

from matplotlib.transforms import Affine2D
import helpers.legacy_hypergrid_graphics as hg

from sklearn.preprocessing import LabelEncoder


from brainblocks.tools import HyperGridTransform
from brainblocks.metrics import gnome_similarity
from brainblocks.datasets import time_series as datasets



import seaborn as sns


def encode_1D_basis(X, ref_points, n_subspace_dims=1, n_grids=1, n_bins=8):
    n_input_dims = X.shape[1]

    custom_bases = np.array([np.identity(n_input_dims)[:n_subspace_dims, :] for k in range(n_grids)])

    # n_input_dims = 3
    # n_grids = 2
    # n_subspace_dims = 2
    """
    custom_bases = np.array([
        [
            [1., 0., 0.],
            [0., 1., 0.]
        ],
        [
            [0., 1., 0.],
            [0., 0., 1.]
        ]
    ])

    custom_periods = np.array([
        [1.0, 2.0],
        [2.0, 1.5]
    ])
    """

    # n_input_dims = 1
    # n_grids = 1
    # n_subspace_dims = 1

    """
    custom_bases = np.array([
        [
            [1.]
        ],
        [
            [1.]
        ]
    ])

    print(custom_bases.shape)
    """
    # print(custom_bases.shape)

    # custom_periods = np.array([
    #    [1.0],
    #    [1.5]
    # ])

    # custom_periods = np.ones((n_grids, n_input_dims))

    # subspace_vector_bases shape: (n_grids, n_subspace_dims, n_input_dims)
    # n_grids = custom_bases.shape[0]
    # n_subspace_dims = custom_bases.shape[1]
    # n_input_dims = custom_bases.shape[2]

    configs = {"n_input_dims": n_input_dims, "n_grids": n_grids, "n_bins": n_bins, "n_acts": 1,
               "n_subspace_dims": n_subspace_dims,
               "flatten_output": True, "max_period": 2.0, "min_period": 0.05,
               "use_normal_dist_bases": False,
               "use_standard_bases": False,
               "use_orthogonal_bases": False,
               "set_bases": custom_bases,
               # "use_random_uniform_periods": True,
               # "set_periods": custom_periods
               }

    # configs["origin"] = tuple([random.uniform(-1.0, 1.0) for k in range(configs["n_input_dims"])])
    configs["origin"] = tuple([0.0 for k in range(configs["n_input_dims"])])

    # instantiate Grid Encoder with initial values
    gridTransform = HyperGridTransform(**configs)

    ref_gnomes = gridTransform.transform(ref_points)
    X_gnomes = gridTransform.transform(X)

    return X_gnomes, ref_gnomes, gridTransform



def plot_2D_hypergrid_illustration_1():

    filename_str = "hypergrid_2D_illustration_1.png"

    #xs, ys = spiral(100)
    #xs, ys = datasets.circle(n=1000, r=1.5)
    xs, ys = datasets.lemiscate(n=1000, alpha=1.2)
    ys = 3 * ys

    num_points = len(xs)
    fig = plt.figure(num=1, figsize=(10, 8))  # , constrained_layout=True)

    fig, ax = plt.subplots(1, 1, num=1)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    random.seed(6)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # for i in range(num_steps):
    #    x_new, y_new = trajectory(xs[i], ys[i])
    #    xs[i + 1] = x_new
    #    ys[i + 1] = y_new

    ax.plot(xs, ys, lw=0.5)
    # print(xs)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    #points = [(xs[point_index], ys[point_index])]
    #points = [(0, 0), (-0.6, -0.8), (0.75, 1.2), (-0.5, 1.1)]
    points = [(0, 0), (-0.6, -0.8)]
    for i, pnt in enumerate(points):
        ax.scatter(pnt[0], pnt[1], color=colors[i])

    periods1 = [0.7, 1.35]
    periods2 = [0.7, 1.35, 1.7]
    periods3 = [0.3, 0.7, 1.35]
    periods4 = [0.3, 0.5, 0.8, 1.1]

    grid_range = 1.0
    hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=0, points=points, periods=periods1, colors=colors, grid_range=grid_range)
    #hg.add_1D_hypergrid(ax, (0, 0), 3.2, angle=45, points=points, periods=periods1, colors=colors, grid_range=grid_range)
    #hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=90, points=points, periods=periods2, colors=colors, grid_range=grid_range)
    #hg.add_1D_hypergrid(ax, (0, 0), 3.2, angle=135, points=points, periods=periods2, colors=colors, grid_range=grid_range)
    #hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=180, points=points, periods=periods3, colors=colors, grid_range=grid_range)
    #hg.add_1D_hypergrid(ax, (0, 0), 3.2, angle=225, points=points, periods=periods3, colors=colors, grid_range=grid_range)
    #hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=270, points=points, periods=periods4, aligned_text=True, colors=colors, grid_range=grid_range)
    #hg.add_1D_hypergrid(ax, (0, 0), 3.2, angle=315, points=points, periods=periods4, aligned_text=True, colors=colors, grid_range=grid_range)

    # plt.show()
    plt.savefig(filename_str) #, bbox_inches='tight')
    plt.clf()

    plt.close()



def plot_2D_hypergrid_frames_1():


    # numPoints = 5
    # points = []
    # for i in range(0, numPoints):
    #    pnt = (random.uniform(-2, 2), random.uniform(-2, 2))
    #    points.append((pnt[0], pnt[1]))

    #dt = 0.01
    #num_steps = 10000

    # Need one more for the initial values
    #xs = np.empty(num_steps + 1)
    #ys = np.empty(num_steps + 1)

    # Set initial values
    #xs[0], ys[0] = (0., 1.)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    # for i in range(num_steps):
    #    x_dot, y_dot = lorenz(xs[i], ys[i])
    #    xs[i + 1] = xs[i] + (x_dot * dt)
    #    ys[i + 1] = ys[i] + (y_dot * dt)

    # df = trajectory(Clifford, 0, 0, -1.3, -1.3, -1.8, -1.9)
    # df = trajectory(Clifford, 0, 0, 1.9, b=1.9, c=1.9, d=0.8)
    # df = random_walk()
    # xs = df['x'].values
    # xs *= 10.0
    # ys = df['y'].values
    # ys *= 10.0

    filename_str = "out/hypergrid_2D_frames_1_%05u.png"


    #xs, ys = spiral(100)
    #xs, ys = datasets.circle(n=100, r=1.5)

    xs, ys = datasets.lemiscate(n=100, alpha=1.2)
    ys = 3 * ys

    num_points = len(xs)
    fig = plt.figure(num=1, figsize=(10, 8))  # , constrained_layout=True)

    #fig.subplots_adjust(left=0.25, right=0.75, bottom=0.1, top=0.75)

    for point_index in range(num_points):

        fig, ax = plt.subplots(1, 1, num=1)
        #fig.subplots_adjust(left=0.25, right=0.75, bottom=0.15, top=0.75)
        fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
        random.seed(6)
        ax.set_aspect('equal')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        # for i in range(num_steps):
        #    x_new, y_new = trajectory(xs[i], ys[i])
        #    xs[i + 1] = x_new
        #    ys[i + 1] = y_new

        #ax.plot(xs, ys, lw=0.5)
        # print(xs)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        #points = [(0,0), (xs[point_index], ys[point_index]), (-0.6, -0.8), (0.75, 1.2), (-0.5, 1.1)]
        #points = [(0,0), (xs[point_index], ys[point_index])]
        points = [(xs[point_index], ys[point_index])]
        #points = [(0, 0), (-0.6, -0.8), (0.75, 1.2), (-0.5, 1.1)]
        for i, pnt in enumerate(points):
            ax.scatter(pnt[0], pnt[1], color=colors[i])

        periods1 = [0.7, 1.35]
        periods2 = [0.7, 1.35, 1.7]
        periods3 = [0.3, 0.7, 1.35]
        periods4 = [0.3, 0.5, 0.8, 1.1]

        #hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=0, points=points, periods=periods1, colors=colors)
        #hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=90, points=points, periods=periods2, colors=colors)
        #hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=180, points=points, periods=periods3, colors=colors)
        #hg.add_1D_hypergrid(ax, (0, 0), 3, angle=290, points=points, periods=periods4, aligned_text=True, colors=colors)
        #hg.add_1D_hypergrid(ax, (0, 0), 3.7, angle=45, points=points, periods=periods2, colors=colors)
        hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=0, points=points, periods=periods1, colors=colors)
        hg.add_1D_hypergrid(ax, (0, 0), 3.2, angle=45, points=points, periods=periods1, colors=colors)
        hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=90, points=points, periods=periods2, colors=colors)
        hg.add_1D_hypergrid(ax, (0, 0), 3.2, angle=135, points=points, periods=periods2, colors=colors)
        hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=180, points=points, periods=periods3, colors=colors)
        hg.add_1D_hypergrid(ax, (0, 0), 3.2, angle=225, points=points, periods=periods3, colors=colors)
        hg.add_1D_hypergrid(ax, (0, 0), 2.3, angle=270, points=points, periods=periods4, aligned_text=True, colors=colors)
        hg.add_1D_hypergrid(ax, (0, 0), 3.2, angle=315, points=points, periods=periods4, aligned_text=True, colors=colors)


        # plt.show()
        plt.savefig(filename_str % point_index)#, bbox_inches='tight')
        plt.clf()

    plt.close()

    # sns.set()

    # Generate a blue-white-red palette:
    # color_list = sns.diverging_palette(240, 10, n=9)

    # Generate a brighter green-white-purple palette
    # color_list = sns.diverging_palette(150, 275, s=80, l=55, n=9)

    # Generate a blue-black-red palette
    # color_list = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark")

    # Generate a colormap object
    # cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

    # sns.reset_defaults()


def plot_grid_illustration_5():
    filename_str = "hypergrid_illustration_5.png"
    title_str = "1D HyperGrid Example"

    # fig = plt.figure(figsize=(8, 4))
    # fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

    # ax1, aux_ax1 = setup_axes1(fig, 131)
    # aux_ax1.bar([0, 1, 2, 3], [3, 2, 1, 3])

    axes_label_pos = (1.15, 0.5)
    x_max = 1.0
    x_min = -1.0

    # create the subplots with the previously created figure "1"
    # figure = plt.figure(num=1, figsize=(11, 5), constrained_layout=True)
    figure = plt.figure(num=1, figsize=(11, 11), constrained_layout=True)
    f, axes = plt.subplots(4, 1, num=1)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # X_gnomes, ref_gnomes, scores, X, ref_points, y, le, data_name, config = result_per_config

    # reference points for comparison
    # ref_points = np.array([[-4.0], [-1.5], [1.5], [4.0]])
    # ref_points = np.array([[-0.25], [0.5]])
    ref_points = np.array([[0.0]])

    # HGT constants
    min_period = 0.05
    n_grids = 2
    n_subspace_dims = 1
    n_bins = 4

    # input data
    inputs = []
    for index_value in np.arange(x_min, x_max, 0.06):
        inputs.append([index_value])
    X = np.array(inputs)

    # class labels
    class_indices = []
    for pnt in ref_points:
        # find index of minimum difference to reference points
        idx = np.abs(np.subtract(X, pnt)).argmin()
        class_indices.append(idx)

    outputs = np.zeros(X.shape)
    for k in range(len(class_indices)):
        outputs[class_indices[k]] = k + 1

    le = LabelEncoder()
    y = le.fit_transform(outputs.flatten())

    X_gnomes, ref_gnomes, gridTransform = encode_1D_basis(X, ref_points, n_bins=n_bins, n_grids=n_grids,
                                                          n_subspace_dims=n_subspace_dims)
    subspace_periods = gridTransform.subspace_periods

    scores = gnome_similarity(X_gnomes, ref_gnomes)

    # axes for this piece of data
    ax = axes[0]
    ax.scatter(X, np.zeros(scores.shape), marker='o')  # , label=list(ref_points))
    # ax.set_title("%d grids / %d bins" % (n_grids, n_bins))
    # ax.set_ylabel("%d grids / %d bins" % (n_grids, n_bins), rotation=0, labelpad=30)
    ax.text(axes_label_pos[0], axes_label_pos[1], "Sampling of\nequidistant points\non the real line.",
            transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center')

    # ax.set_xlim((X[0], X[-1]))
    ax.set_xlim((x_min, x_max))
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks(())
    # ax.set_xticks(())
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='major', labelsize=16)

    for pnt_index in range(len(X)):
        pnt = X[pnt_index]
        ax.axvline(x=pnt[0], ymax=0.5, ymin=-0.5, linewidth=0.5, color=colors[0], label=float(pnt), clip_on=False,
                   zorder=6)
        # str(interval_count % n_bins), transform=ax.transData, fontsize=8)

        xyA = (pnt, 0)
        xyB = (pnt, 0.5)
        con = patches.ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data",
                                      axesA=axes[1], axesB=ax, color="red")
        axes[1].add_artist(con)
    # for pnt_index in range(len(ref_points)):
    #    pnt = ref_points[pnt_index]
    #    ax.axvline(x=pnt[0], ymin=-1, ymax=1, linewidth=1.5, color=colors[pnt_index], label=float(pnt), zorder=6)

    # plot the grid overlayed the binned space
    x_high = x_max
    x_low = x_min
    ax = axes[1]

    mags = list(subspace_periods.reshape(-1))

    for grid_i in range(n_grids):

        # grid magnitude
        mag1 = mags[grid_i]

        # origin grid center offset
        origin_x = -mag1 / (2.0 * n_bins)

        # interval length of bins
        bin_interval_x = mag1 / n_bins

        # vertical position for grid in the plot
        grid_vpos = (grid_i / n_grids, (grid_i + 1.0) / n_grids)

        text_offset = -0.01

        interval_count = 0
        while interval_count * bin_interval_x <= x_high + bin_interval_x:
            bin_boundary_x = origin_x + bin_interval_x * interval_count
            ax.axvline(x=bin_boundary_x, ymin=grid_vpos[0], ymax=grid_vpos[1], linewidth=1.5, color='k')

            ax.text(bin_boundary_x + bin_interval_x * 0.5 + text_offset, grid_vpos[0] + 0.4 / n_grids,
                    str(interval_count % n_bins), transform=ax.transData, fontsize=8, clip_on=True)

            interval_count += 1

        interval_count = -1
        while interval_count * bin_interval_x >= x_low - bin_interval_x:
            bin_boundary_x = origin_x + bin_interval_x * interval_count
            ax.axvline(x=bin_boundary_x, ymin=grid_vpos[0], ymax=grid_vpos[1], linewidth=1.5, color='k')
            ax.text(bin_boundary_x + bin_interval_x * 0.5 + text_offset, grid_vpos[0] + 0.4 / n_grids,
                    str(interval_count % n_bins), transform=ax.transData, fontsize=8, clip_on=True)
            interval_count -= 1

        ax.axhline(grid_vpos[1], lw=1.5, color='k')
        ax.axhline(grid_vpos[0], lw=1.5, color='k')

    # for pnt_index in range(len(X)):
    #    pnt = X[pnt_index]
    #    ax.axvline(x=pnt[0], linewidth=1.5, color=colors[0], label=float(pnt), zorder=1, alpha=0.4)
    # for pnt_index in range(len(ref_points)):
    #    pnt = ref_points[pnt_index]
    #    ax.axvline(x=pnt[0], linewidth=1.5, color=colors[pnt_index], label=float(pnt), zorder=6)

    ax.set_xlim((x_min, x_max))
    # ax.set_ylim((0, n_bins*n_grids*n_subspace_dims/))
    ax.set_ylim(0, 1.0)

    # rest of ticks for y axis
    ax.yaxis.set_major_locator(ticker.IndexLocator(1.0 / n_grids, 0))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # magnitude labels
    labelsizes = [12]
    ax.tick_params(axis='y', which='minor', labelsize=labelsizes[0])
    ax.tick_params(axis='y', which='major', length=25, direction='out')
    ax.tick_params(axis='y', which='minor', length=0)
    mags = list(subspace_periods.reshape(-1))
    mags = ["%.2f" % mag for mag in mags]
    # mags.reverse()
    ax.yaxis.set_minor_locator(ticker.IndexLocator(1.0 / n_grids, 0.5 / n_grids))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))

    ax.set_xticks(())
    ax.text(axes_label_pos[0], axes_label_pos[1],
            "Cyclical binning of space.\n%d grids, %d bins/grid.\nGrid period shown on y." % (n_grids, n_bins),
            transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center')

    # axes for this piece of data
    ax = axes[2]

    # for k in range(len(y)):
    #    if y[k] != 0:
    #        ax.axvline(x=float(float(k) + 0.5), linewidth=1.5, color=colors[y[k] - 1],
    #                   label=float(ref_points[y[k] - 1]), zorder=6)

    cmap = ListedColormap(['white', 'gray'])

    # barprops = dict(cmap='binary', interpolation='nearest', aspect='auto')
    state_data = np.rot90(X_gnomes, k=-1, axes=(1, 0))
    barprops = dict(cmap=cmap, interpolation='nearest', aspect='auto',
                    extent=[0, state_data.shape[1], 0, state_data.shape[0]])
    img = ax.imshow(state_data, **barprops)
    # img = ax.matshow(state_data, **barprops)

    for y in range(state_data.shape[0] + 1):
        ax.axhline(y, lw=0.5, color='k', zorder=5)
    for x in range(state_data.shape[1] + 1):
        ax.axvline(x, lw=0.5, color='k', zorder=5)

    # disable x ticks
    ax.xaxis.set_major_locator(ticker.NullLocator())

    # rest of ticks for y axis
    ax.yaxis.set_major_locator(ticker.IndexLocator(n_bins, 0))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # magnitude labels

    ax.tick_params(axis='y', which='minor', labelsize=labelsizes[0])
    ax.tick_params(axis='y', which='major', length=25, direction='out')
    ax.tick_params(axis='y', which='minor', length=0)
    mags = list(subspace_periods.reshape(-1))
    mags = ["%.2f" % mag for mag in mags]
    # mags.reverse()
    ax.yaxis.set_minor_locator(ticker.IndexLocator(n_bins, n_bins // 2))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))

    ax.set_xlim((0.5, state_data.shape[1] - 0.5))

    ax.text(axes_label_pos[0], axes_label_pos[1],
            "Encoding for each\nsampled point.\nSection by grid\nperiod shown on y", transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center')

    # for x in range(state_data.shape[1] + 1):
    #    ax.axvline(x=x+0.5, ymin=0, ymax=1, linewidth=1.5, color=colors[0], zorder=1, alpha=0.4)

    # axes for this piece of data
    ax = axes[3]
    ax.plot(X, scores, marker='o')  # , label=list(ref_points))
    # ax.fill_between(X.flatten(), scores.flatten(), -1.0)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(axis='x', which='major', labelsize=16)

    ax.text(axes_label_pos[0], axes_label_pos[1],
            "Cosine similarity of\nencoded points to\nreference point %.1f." % (float(ref_points)),
            transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center')

    plt.vlines(X, scores, 2.0, linewidth=0.5, color=colors[0])

    # for pnt_index in range(len(X)):
    #    pnt = X[pnt_index]
    #    ax.axvline(x=pnt[0], linewidth=0.5, color=colors[0], label=float(pnt), zorder=6)

    # create the legend on the figure, not the axes
    # handles, labels = axes[1].get_legend_handles_labels()
    # figure.legend(handles, labels, title="Comparison\nPoints", loc='upper right',
    #              bbox_to_anchor=(1.1, 0.9))  # , fontsize=20)

    # figure title
    figure.suptitle(title_str + "\n")

    # print("figure properties")
    # print(getp(figure))
    # print("axes properties")
    # print(getp(ax))

    # save figure plot to file
    plt.savefig(filename_str, bbox_inches='tight')
    plt.close()


def plot_grid_illustration_4():
    filename_str = "hypergrid_illustration_4.png"
    title_str = "1D HyperGrid Example"

    axes_label_pos = (1.15, 0.5)
    x_max = 1.0
    x_min = -1.0

    # create the subplots with the previously created figure "1"
    figure = plt.figure(num=1, figsize=(11, 11), constrained_layout=True)
    f, axes = plt.subplots(4, 1, num=1)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # reference points for comparison
    ref_points = np.array([[0.0]])

    # HGT constants
    min_period = 0.05
    n_grids = 2
    n_subspace_dims = 1
    n_bins = 4

    # input data
    inputs = []
    for index_value in np.arange(x_min, x_max, 0.06):
        inputs.append([index_value])
    X = np.array(inputs)

    # class labels
    class_indices = []
    for pnt in ref_points:
        # find index of minimum difference to reference points
        idx = np.abs(np.subtract(X, pnt)).argmin()
        class_indices.append(idx)

    outputs = np.zeros(X.shape)
    for k in range(len(class_indices)):
        outputs[class_indices[k]] = k + 1

    le = LabelEncoder()
    y = le.fit_transform(outputs.flatten())

    X_gnomes, ref_gnomes, gridTransform = encode_1D_basis(X, ref_points, n_bins=n_bins, n_grids=n_grids,
                                                          n_subspace_dims=n_subspace_dims)
    subspace_periods = gridTransform.subspace_periods

    scores = gnome_similarity(X_gnomes, ref_gnomes)

    # axes for this piece of data
    ax = axes[0]
    ax.scatter(X, np.zeros(scores.shape), marker='o')  # , label=list(ref_points))
    ax.text(axes_label_pos[0], axes_label_pos[1], "Sampling of\nequidistant points\non the real line.",
            transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center')

    ax.set_xlim((x_min, x_max))
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks(())
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='major', labelsize=16)

    for pnt_index in range(len(X)):
        pnt = X[pnt_index]
        ax.axvline(x=pnt[0], ymax=0.5, ymin=0.0, linewidth=0.5, color=colors[0], label=float(pnt))  # , zorder=6)

    # plot the grid overlayed the binned space
    x_high = x_max
    x_low = x_min
    ax = axes[1]

    mags = list(subspace_periods.reshape(-1))

    for grid_i in range(n_grids):

        # grid magnitude
        mag1 = mags[grid_i]

        # origin grid center offset
        origin_x = -mag1 / (2.0 * n_bins)

        # interval length of bins
        bin_interval_x = mag1 / n_bins

        # vertical position for grid in the plot
        grid_vpos = (grid_i / n_grids, (grid_i + 1.0) / n_grids)

        text_offset = -0.01

        interval_count = 0
        while interval_count * bin_interval_x <= x_high + bin_interval_x:
            bin_boundary_x = origin_x + bin_interval_x * interval_count
            ax.axvline(x=bin_boundary_x, ymin=grid_vpos[0], ymax=grid_vpos[1], linewidth=1.5, color='k')

            ax.text(bin_boundary_x + bin_interval_x * 0.5 + text_offset, grid_vpos[0] + 0.4 / n_grids,
                    str(interval_count % n_bins), transform=ax.transData, fontsize=8, clip_on=True)

            interval_count += 1

        interval_count = -1
        while interval_count * bin_interval_x >= x_low - bin_interval_x:
            bin_boundary_x = origin_x + bin_interval_x * interval_count
            ax.axvline(x=bin_boundary_x, ymin=grid_vpos[0], ymax=grid_vpos[1], linewidth=1.5, color='k')
            ax.text(bin_boundary_x + bin_interval_x * 0.5 + text_offset, grid_vpos[0] + 0.4 / n_grids,
                    str(interval_count % n_bins), transform=ax.transData, fontsize=8, clip_on=True)
            interval_count -= 1

        ax.axhline(grid_vpos[1], lw=1.5, color='k')
        ax.axhline(grid_vpos[0], lw=1.5, color='k')

    ax.set_xlim((x_min, x_max))
    ax.set_ylim(0, 1.0)

    # rest of ticks for y axis
    ax.yaxis.set_major_locator(ticker.IndexLocator(1.0 / n_grids, 0))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # magnitude labels
    labelsizes = [12]
    ax.tick_params(axis='y', which='minor', labelsize=labelsizes[0])
    ax.tick_params(axis='y', which='major', length=25, direction='out')
    ax.tick_params(axis='y', which='minor', length=0)
    mags = list(subspace_periods.reshape(-1))
    mags = ["%.2f" % mag for mag in mags]

    ax.yaxis.set_minor_locator(ticker.IndexLocator(1.0 / n_grids, 0.5 / n_grids))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))

    ax.set_xticks(())
    ax.text(axes_label_pos[0], axes_label_pos[1],
            "Cyclical binning of space.\n%d grids, %d bins/grid.\nGrid period shown on y." % (n_grids, n_bins),
            transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center')

    # axes for this piece of data
    ax = axes[2]

    cmap = ListedColormap(['white', 'gray'])

    # barprops = dict(cmap='binary', interpolation='nearest', aspect='auto')
    state_data = np.rot90(X_gnomes, k=-1, axes=(1, 0))
    barprops = dict(cmap=cmap, interpolation='nearest', aspect='auto',
                    extent=[0, state_data.shape[1], 0, state_data.shape[0]])
    img = ax.imshow(state_data, **barprops)

    for y in range(state_data.shape[0] + 1):
        ax.axhline(y, lw=0.5, color='k', zorder=5)
    for x in range(state_data.shape[1] + 1):
        ax.axvline(x, lw=0.5, color='k', zorder=5)

    # disable x ticks
    ax.xaxis.set_major_locator(ticker.NullLocator())

    # rest of ticks for y axis
    ax.yaxis.set_major_locator(ticker.IndexLocator(n_bins, 0))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # magnitude labels

    ax.tick_params(axis='y', which='minor', labelsize=labelsizes[0])
    ax.tick_params(axis='y', which='major', length=25, direction='out')
    ax.tick_params(axis='y', which='minor', length=0)
    mags = list(subspace_periods.reshape(-1))
    mags = ["%.2f" % mag for mag in mags]

    ax.yaxis.set_minor_locator(ticker.IndexLocator(n_bins, n_bins // 2))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))

    ax.set_xlim((0.5, state_data.shape[1] - 0.5))

    ax.text(axes_label_pos[0], axes_label_pos[1],
            "Encoding for each\nsampled point.\nSection by grid\nperiod shown on y", transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center')

    # axes for this piece of data
    ax = axes[3]
    ax.plot(X, scores, marker='o')
    ax.set_xlim((x_min, x_max))
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(axis='x', which='major', labelsize=16)

    ax.text(axes_label_pos[0], axes_label_pos[1],
            "Cosine similarity of\nencoded points to\nreference point %.1f." % (float(ref_points)),
            transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center')

    plt.vlines(X, scores, 2.0, linewidth=0.5, color=colors[0])

    # figure title
    figure.suptitle(title_str + "\n")

    # save figure plot to file
    plt.savefig(filename_str, bbox_inches='tight')
    plt.close()


def plot_grid_illustration_2():
    filename_str = "hypergrid_illustration_2.png"
    title_str = "1D HyperGrid Example"

    # create the subplots with the previously created figure "1"
    figure = plt.figure(num=1, figsize=(11, 7), constrained_layout=True)
    f, axes = plt.subplots(2, 1, num=1)

    # reference points for comparison
    # ref_points = np.array([[-4.0], [-1.5], [1.5], [4.0]])
    ref_points = np.array([[-0.25], [0.5]])

    # HGT constants
    min_period = 0.05
    n_grids = 8
    n_subspace_dims = 1
    n_bins = 8

    # input data
    inputs = []
    for index_value in np.arange(-1.0, 1.0, 0.025):
        inputs.append([index_value])
    X = np.array(inputs)

    # class labels
    class_indices = []
    for pnt in ref_points:
        # find index of minimum difference to reference points
        idx = np.abs(np.subtract(X, pnt)).argmin()
        class_indices.append(idx)

    outputs = np.zeros(X.shape)
    for k in range(len(class_indices)):
        outputs[class_indices[k]] = k + 1

    le = LabelEncoder()
    y = le.fit_transform(outputs.flatten())

    X_gnomes, ref_gnomes, gridTransform = encode_1D_basis(X, ref_points, n_bins=n_bins, n_grids=n_grids,
                                                          n_subspace_dims=n_subspace_dims)
    subspace_periods = gridTransform.subspace_periods

    scores = gnome_similarity(X_gnomes, ref_gnomes)

    # axes for this piece of data
    ax = axes[0]
    ax.plot(X, scores)  # , label=list(ref_points))
    ax.set_title("%d grids / %d bins" % (n_grids, n_bins))
    ax.set_xlim((X[0], X[-1]))

    # axes for this piece of data
    ax = axes[1]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for k in range(len(y)):
        if y[k] != 0:
            ax.axvline(x=float(float(k) + 0.5), linewidth=1.5, color=colors[y[k] - 1],
                       label=float(ref_points[y[k] - 1]), zorder=6)

    cmap = ListedColormap(['white', 'gray'])

    state_data = np.rot90(X_gnomes, k=-1, axes=(1, 0))
    barprops = dict(cmap=cmap, interpolation='nearest', aspect='auto',
                    extent=[0, state_data.shape[1], 0, state_data.shape[0]])
    img = ax.imshow(state_data, **barprops)

    for y in range(state_data.shape[0] + 1):
        ax.axhline(y, lw=0.5, color='k', zorder=5)
    for x in range(state_data.shape[1] + 1):
        ax.axvline(x, lw=0.5, color='k', zorder=5)

    # disable x ticks
    ax.xaxis.set_major_locator(ticker.NullLocator())

    # rest of ticks for y axis
    ax.yaxis.set_major_locator(ticker.IndexLocator(n_bins, 0))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # magnitude labels
    labelsizes = [12]
    ax.tick_params(axis='y', which='minor', labelsize=labelsizes[0])
    ax.tick_params(axis='y', which='major', length=25, direction='out')
    ax.tick_params(axis='y', which='minor', length=0)
    mags = list(subspace_periods.reshape(-1))
    mags = ["%.2f" % mag for mag in mags]

    ax.yaxis.set_minor_locator(ticker.IndexLocator(n_bins, n_bins // 2))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))

    # create the legend on the figure, not the axes
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, title="Comparison\nPoints", loc='upper right',
                  bbox_to_anchor=(1.1, 0.9))  # , fontsize=20)

    # figure title
    figure.suptitle(title_str + "\n")

    # save figure plot to file
    plt.savefig(filename_str, bbox_inches='tight')
    plt.close()


def plot_grid_illustration_3():
    filename_str = "hypergrid_illustration_3.png"
    title_str = "1D HyperGrid Example"

    # create the subplots with the previously created figure "1"
    figure = plt.figure(num=1, figsize=(11, 7), constrained_layout=True)
    f, axes = plt.subplots(3, 1, num=1)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # reference points for comparison
    ref_points = np.array([[-0.25], [0.5]])

    # HGT constants
    min_period = 0.05
    n_grids = 4
    n_subspace_dims = 1
    n_bins = 4

    # input data
    inputs = []
    for index_value in np.arange(-1.0, 1.0, 0.06):
        inputs.append([index_value])
    X = np.array(inputs)

    # class labels
    class_indices = []
    for pnt in ref_points:
        # find index of minimum difference to reference points
        idx = np.abs(np.subtract(X, pnt)).argmin()
        class_indices.append(idx)

    outputs = np.zeros(X.shape)
    for k in range(len(class_indices)):
        outputs[class_indices[k]] = k + 1

    le = LabelEncoder()
    y = le.fit_transform(outputs.flatten())

    X_gnomes, ref_gnomes, gridTransform = encode_1D_basis(X, ref_points, n_bins=n_bins, n_grids=n_grids,
                                                          n_subspace_dims=n_subspace_dims)
    subspace_periods = gridTransform.subspace_periods

    scores = gnome_similarity(X_gnomes, ref_gnomes)

    # axes for this piece of data
    ax = axes[0]
    ax.plot(X, scores, marker='o')  # , label=list(ref_points))
    ax.set_title("%d grids / %d bins" % (n_grids, n_bins))
    ax.set_xlim((X[0], X[-1]))
    ax.set_ylim(-0.1, 1.1)

    for pnt_index in range(len(ref_points)):
        pnt = ref_points[pnt_index]
        ax.axvline(x=pnt[0], linewidth=1.5, color=colors[pnt_index], label=float(pnt), zorder=6)

    # plot the grid overlayed the binned space
    x_range = float(X[-1] - X[0])
    x_high = float(X[-1])
    x_low = float(X[0])
    ax = axes[1]

    mags = list(subspace_periods.reshape(-1))

    for grid_i in range(n_grids):

        # grid magnitude
        mag1 = mags[grid_i]

        # origin grid center offset
        origin_x = -mag1 / (2.0 * n_bins)

        # interval length of bins
        bin_interval_x = mag1 / n_bins

        # vertical position for grid in the plot
        grid_vpos = (grid_i / n_grids, (grid_i + 1.0) / n_grids)

        text_offset = -0.01

        interval_count = 0
        while interval_count * bin_interval_x < x_high:
            bin_boundary_x = origin_x + bin_interval_x * interval_count
            ax.axvline(x=bin_boundary_x, ymin=grid_vpos[0], ymax=grid_vpos[1], linewidth=1.5, color='k')

            ax.text(bin_boundary_x + bin_interval_x * 0.5 + text_offset, grid_vpos[0] + 0.4 / n_grids,
                    str(interval_count % n_bins), transform=ax.transData, fontsize=8)

            interval_count += 1

        interval_count = -1
        while interval_count * bin_interval_x > x_low:
            bin_boundary_x = origin_x + bin_interval_x * interval_count
            ax.axvline(x=bin_boundary_x, ymin=grid_vpos[0], ymax=grid_vpos[1], linewidth=1.5, color='k')
            ax.text(bin_boundary_x + bin_interval_x * 0.5 + text_offset, grid_vpos[0] + 0.4 / n_grids,
                    str(interval_count % n_bins), transform=ax.transData, fontsize=8)
            interval_count -= 1

        ax.axhline(grid_vpos[1], lw=1.5, color='k')
        ax.axhline(grid_vpos[0], lw=1.5, color='k')

    for pnt_index in range(len(ref_points)):
        pnt = ref_points[pnt_index]
        ax.axvline(x=pnt[0], linewidth=1.5, color=colors[pnt_index], label=float(pnt), zorder=6)

    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0, 1.0)

    # rest of ticks for y axis
    ax.yaxis.set_major_locator(ticker.IndexLocator(1.0 / n_grids, 0))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # magnitude labels
    labelsizes = [12]
    ax.tick_params(axis='y', which='minor', labelsize=labelsizes[0])
    ax.tick_params(axis='y', which='major', length=25, direction='out')
    ax.tick_params(axis='y', which='minor', length=0)
    mags = list(subspace_periods.reshape(-1))
    mags = ["%.2f" % mag for mag in mags]
    ax.yaxis.set_minor_locator(ticker.IndexLocator(1.0 / n_grids, 0.5 / n_grids))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))

    # axes for this piece of data
    ax = axes[2]

    for k in range(len(y)):
        if y[k] != 0:
            ax.axvline(x=float(float(k) + 0.5), linewidth=1.5, color=colors[y[k] - 1],
                       label=float(ref_points[y[k] - 1]), zorder=6)

    cmap = ListedColormap(['white', 'gray'])

    state_data = np.rot90(X_gnomes, k=-1, axes=(1, 0))
    barprops = dict(cmap=cmap, interpolation='nearest', aspect='auto',
                    extent=[0, state_data.shape[1], 0, state_data.shape[0]])
    img = ax.imshow(state_data, **barprops)

    for y in range(state_data.shape[0] + 1):
        ax.axhline(y, lw=0.5, color='k', zorder=5)
    for x in range(state_data.shape[1] + 1):
        ax.axvline(x, lw=0.5, color='k', zorder=5)

    # disable x ticks
    ax.xaxis.set_major_locator(ticker.NullLocator())

    # rest of ticks for y axis
    ax.yaxis.set_major_locator(ticker.IndexLocator(n_bins, 0))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # magnitude labels
    labelsizes = [12]
    ax.tick_params(axis='y', which='minor', labelsize=labelsizes[0])
    ax.tick_params(axis='y', which='major', length=25, direction='out')
    ax.tick_params(axis='y', which='minor', length=0)
    mags = list(subspace_periods.reshape(-1))
    mags = ["%.2f" % mag for mag in mags]
    # mags.reverse()
    ax.yaxis.set_minor_locator(ticker.IndexLocator(n_bins, n_bins // 2))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))

    ax.set_xlim((0.5, state_data.shape[1] - 0.5))

    # create the legend on the figure, not the axes
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, title="Comparison\nPoints", loc='upper right',
                  bbox_to_anchor=(1.1, 0.9))  # , fontsize=20)

    # figure title
    figure.suptitle(title_str + "\n")

    # save figure plot to file
    plt.savefig(filename_str, bbox_inches='tight')
    plt.close()


def plot_grid_illustration_1():
    filename_str = "hypergrid_illustration_1.png"
    title_str = "1D HyperGrid Example"

    # create the subplots with the previously created figure "1"
    figure = plt.figure(num=1, figsize=(11, 7), constrained_layout=True)
    f, axes = plt.subplots(2, 1, num=1)

    # reference points for comparison
    ref_points = np.array([[-4.0], [1.5]])

    # HGT constants
    min_period = 0.05
    n_grids = 1
    n_subspace_dims = 1
    n_bins = 8

    # input data
    inputs = []
    for index_value in np.arange(-5.0, 5.0, min_period):
        inputs.append([index_value])
    X = np.array(inputs)

    # class labels
    class_indices = []
    for pnt in ref_points:
        # find index of minimum difference to reference points
        idx = np.abs(np.subtract(X, pnt)).argmin()
        class_indices.append(idx)

    outputs = np.zeros(X.shape)
    for k in range(len(class_indices)):
        outputs[class_indices[k]] = k + 1

    le = LabelEncoder()
    y = le.fit_transform(outputs.flatten())

    X_gnomes, ref_gnomes, gridTransform = encode_1D_basis(X, ref_points, n_bins=n_bins, n_grids=n_grids,
                                                          n_subspace_dims=n_subspace_dims)
    subspace_periods = gridTransform.subspace_periods

    scores = gnome_similarity(X_gnomes, ref_gnomes)

    # axes for this piece of data
    ax = axes[0]
    ax.plot(X, scores)  # , label=list(ref_points))
    ax.set_title("%d grids / %d bins" % (n_grids, n_bins))
    ax.set_xlim((X[0], X[-1]))

    # axes for this piece of data
    ax = axes[1]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for k in range(len(y)):
        if y[k] != 0:
            ax.axvline(x=k, linewidth=1.5, color=colors[y[k] - 1],
                       label=float(ref_points[y[k] - 1]))

    state_data = np.rot90(X_gnomes, k=-1, axes=(1, 0))
    barprops = dict(cmap='binary', interpolation='nearest', aspect='auto',
                    extent=[0, state_data.shape[1], 0, state_data.shape[0]])
    img = ax.imshow(state_data, **barprops)

    # disable x ticks
    ax.xaxis.set_major_locator(ticker.NullLocator())

    # rest of ticks for y axis
    ax.yaxis.set_major_locator(ticker.IndexLocator(n_bins, 0))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # magnitude labels
    labelsizes = [12]
    ax.tick_params(axis='y', which='minor', labelsize=labelsizes[0])
    ax.tick_params(axis='y', which='major', length=25, direction='out')
    ax.tick_params(axis='y', which='minor', length=0)
    mags = list(subspace_periods.reshape(-1))
    mags = ["%.2f" % mag for mag in mags]
    # mags.reverse()
    ax.yaxis.set_minor_locator(ticker.IndexLocator(n_bins, n_bins // 2))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))

    # create the legend on the figure, not the axes
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, title="Comparison\nPoints", loc='upper right',
                  bbox_to_anchor=(1.1, 0.9))  # , fontsize=20)

    # figure title
    figure.suptitle(title_str + "\n")

    # save figure plot to file
    plt.savefig(filename_str, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # plot_grid_illustration_1()
    # plot_grid_illustration_2()
    # plot_grid_illustration_3()
    # plot_grid_illustration_4()
    #plot_grid_illustration_5()
    #plot_2D_hypergrid_frames_1()
    plot_2D_hypergrid_illustration_1()
