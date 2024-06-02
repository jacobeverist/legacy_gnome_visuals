import numpy as np
from copy import deepcopy

import os
import glob
import pprint

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, colorConverter, NoNorm
import matplotlib.patches as patches
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import random

from matplotlib import ticker
from matplotlib.transforms import Affine2D
#import helpers.datasets as datasets
# import helpers.hypergrid_graphics as hg


from brainblocks.datasets import time_series as datasets
from brainblocks.tools import HyperGridTransform
from brainblocks.metrics import gnome_similarity

import seaborn as sns


def create_transform_2D_to_1D_hypergrids(n_input_dims=2, n_subspace_dims=1, n_grids=1, n_bins=4,
                                         periods=None, angles=None):
    # print(custom_bases)
    # print(custom_bases.shape)
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

    # subspace_vector_bases shape: (n_grids, n_subspace_dims, n_input_dims)
    # n_grids = custom_bases.shape[0]
    # n_subspace_dims = custom_bases.shape[1]
    # n_input_dims = custom_bases.shape[2]

    custom_bases = np.array([np.identity(n_input_dims)[:n_subspace_dims, :] for k in range(n_grids)])
    if isinstance(angles, list):
        for i in range(n_grids):
            # angle = 2 * np.pi * i / n_grids
            angle = 2 * np.pi * angles[i] / 360.0
            custom_bases[i][0] = [np.cos(angle), np.sin(angle)]

    custom_periods = np.ones((n_grids, n_subspace_dims))
    if not periods is None:
        for i in range(n_grids):
            for j in range(n_subspace_dims):
                custom_periods[i, j] = periods[i * n_subspace_dims + j]

    # print(custom_periods)

    configs = {"n_input_dims": n_input_dims, "n_grids": n_grids, "n_bins": n_bins, "n_acts": 1,
               "n_subspace_dims": n_subspace_dims,
               "flatten_output": False, "max_period": 2.0, "min_period": 0.05,
               "use_normal_dist_bases": False,
               "use_standard_bases": False,
               "use_orthogonal_bases": False,
               # "set_bases": custom_bases,
               # "use_random_uniform_periods": True,
               # "set_periods": custom_periods
               }

    if isinstance(angles, list):
        configs["set_bases"] = custom_bases

    if not periods is None:
        configs["set_periods"] = custom_periods

    print("HyperGridTransform config:")
    pp = pprint.PrettyPrinter()
    pp.pprint(configs)

    configs["origin"] = tuple([0.0 for k in range(configs["n_input_dims"])])

    # configs["origin"] = tuple([random.uniform(-1.0, 1.0) for k in range(configs["n_input_dims"])])
    # print(n_input_dims, n_grids, n_bins, n_subspace_dims)
    # print(custom_periods)
    # print(custom_periods.shape)

    # instantiate Grid Encoder with initial values
    gridTransform = HyperGridTransform(**configs)

    # ref_gnomes = gridTransform.transform(ref_points)
    # X_gnomes = gridTransform.transform(X)

    # return X_gnomes, ref_gnomes, gridTransform
    return gridTransform


def add_1D_hypergrid(ax, origin, projection_length, points=(), n_bins=4, periods=(0.7, 1.35), angle=0, grid_range=3.0,
                     box_height=0.1,
                     aligned_text=False, colors=None):
    rotation_point = origin

    projection_angle = angle + 90

    # the projected starting point for grid rectangles
    projection_point_rotation = Affine2D().rotate_deg_around(rotation_point[0], rotation_point[1], projection_angle)
    origin_x, origin_y = projection_point_rotation.transform([rotation_point[0] + projection_length, rotation_point[1]])

    n_grids = len(periods)

    x_high = grid_range / 2
    x_low = -grid_range / 2

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    grid_dists = []

    for iter_index, pnt in enumerate(points):
        # projected distance to origin from sample point
        point_to_origin_dist = (origin_x - pnt[0]) * np.cos(projection_angle * np.pi / 180) + (
                origin_y - pnt[1]) * np.sin(projection_angle * np.pi / 180)

        line_x = pnt[0] + np.cos(np.pi / 180.0 * projection_angle) * (point_to_origin_dist + n_grids * box_height)
        line_y = pnt[1] + np.sin(np.pi / 180.0 * projection_angle) * (point_to_origin_dist + n_grids * box_height)

        # ax.plot([pnt[0], line_x], [pnt[1], line_y], clip_on=False, color=colors[iter_index], alpha=0.3)  # , zorder=6)
        ax.plot([pnt[0], line_x], [pnt[1], line_y], clip_on=False, color=colors[iter_index], alpha=0.7)  # , zorder=6)

        grid_lateral_dist = (pnt[0] - rotation_point[0]) * np.cos(angle * np.pi / 180) + (
                pnt[1] - rotation_point[1]) * np.sin(angle * np.pi / 180)

        # print(angle, grid_lateral_dist, pnt)
        grid_dists.append(grid_lateral_dist)

    # data space coordinates to find new point after rotated
    fixed_point_rotation = Affine2D().rotate_deg_around(origin_x, origin_y, angle)

    # plot the grid overlayed the binned space
    for grid_i in range(n_grids):

        # grid magnitude
        period = periods[grid_i]

        # interval length of bins
        bin_interval_x = period / n_bins

        # position of bottom left corner of the first bin rectangle of this grid
        grid_x = origin_x - bin_interval_x / 2
        grid_y = grid_i * box_height + origin_y

        bin_indices = []
        for point_index, dist in enumerate(grid_dists):
            # get value within period
            x_period = (dist + bin_interval_x / 2) % period

            # convert to unit interval
            x_unit = x_period / period

            # indexes for setting bits
            bin_index = np.floor(x_unit * n_bins).astype(int)

            bin_indices.append(bin_index)

        # index of first bin within specified range
        interval_count = int((x_low - bin_interval_x) // bin_interval_x) + 1
        while interval_count * bin_interval_x <= x_high + bin_interval_x:
            # bottom left corner of current bin rectangle
            bin_boundary_x = grid_x + bin_interval_x * interval_count
            bin_boundary_y = grid_y

            # bottom left corner after its been rotated
            corner_pos = fixed_point_rotation.transform([bin_boundary_x, bin_boundary_y])

            num_occs = 0
            if interval_count >= 0 and interval_count < n_bins:
                # total_color = 'gray'
                total_color = 'none'
            else:
                total_color = 'none'

            for point_index, dist in enumerate(grid_dists):
                bin_index = bin_indices[point_index]

                if bin_index == interval_count % n_bins:

                    curr_color = colors[point_index]
                    curr_color = np.array(colorConverter.to_rgba(curr_color))

                    if isinstance(total_color, str):
                        total_color = curr_color
                    else:
                        total_color += curr_color

                    num_occs += 1

            alpha = 1.0
            linewidth = 1.5
            if interval_count < 0:
                # m = abs(interval_count)
                # alpha = (0.5 - m * 0.1)
                m = abs(bin_interval_x * interval_count)
                alpha = 0.3 - m * 0.15
                linewidth = 0.5
            elif interval_count >= n_bins:
                # m = interval_count - n_bins + 1
                # alpha = (0.5 - m * 0.1)
                m = abs(bin_interval_x * interval_count)
                alpha = 0.3 - m * 0.15
                linewidth = 0.5
            if alpha < 0:
                alpha = 0

            """
            
            m = abs(bin_interval_x * interval_count)
            alpha = 1.0 - m * 1
            if alpha < 0:
                alpha = 0
            """

            if num_occs > 0:
                total_color = total_color / num_occs

            # create rectangle with text inside
            add_text_rect(ax, corner_pos[0], corner_pos[1], bin_interval_x, box_height, angle=angle,
                          text_str=str(interval_count % n_bins), aligned_text=aligned_text, facecolor=total_color,
                          alpha=alpha, linewidth=linewidth)

            # increment counter
            interval_count += 1

        """
        interval_count = 0
        while interval_count < n_bins:
            # bottom left corner of current bin rectangle
            bin_boundary_x = grid_x + bin_interval_x * interval_count
            bin_boundary_y = grid_y

            # bottom left corner after its been rotated
            corner_pos = fixed_point_rotation.transform([bin_boundary_x, bin_boundary_y])

            num_occs = 0
            total_color = 'none'
            for point_index, dist in enumerate(grid_dists):
                bin_index = bin_indices[point_index]

                if bin_index == interval_count % n_bins:

                    curr_color = colors[point_index]
                    curr_color = np.array(colorConverter.to_rgba(curr_color))

                    if isinstance(total_color, str):
                        total_color = curr_color
                    else:
                        total_color += curr_color

                    num_occs += 1

            if num_occs > 0:
                total_color = total_color / num_occs

            # create rectangle with text inside
            add_text_rect(ax, corner_pos[0], corner_pos[1], bin_interval_x, box_height, angle=angle,
                          text_str=str(interval_count % n_bins), aligned_text=aligned_text, facecolor=total_color)

            # increment counter
            interval_count += 1
        """

    # regularize angle between +/- 180
    text_angle = angle
    if angle > 0:
        while text_angle > 180:
            text_angle -= 360
    elif angle < 0:
        while text_angle < -180:
            text_angle += 360

    pos_angle = text_angle
    if pos_angle < 0:
        pos_angle += 360

    if aligned_text:
        transform_angle = text_angle
        if abs(text_angle) > 90:
            transform_angle += 180
    else:
        transform_angle = 0

    # label for the grid projection bins
    label_padding = 1.5 * box_height
    label_x, label_y = projection_point_rotation.transform(
        [rotation_point[0] + projection_length + n_grids * box_height + label_padding, rotation_point[1]])

    #ax.text(label_x, label_y, "%.0f$^\circ$" % pos_angle, rotation=transform_angle, rotation_mode='anchor',
    #        fontsize=16, va='center', ha='center', clip_on=False)


def add_text_rect(ax, box_x, box_y, box_width, box_height, angle=0, linewidth=1.5, edgecolor='k', fontsize=8,
                  facecolor='none', text_str=None, aligned_text=False, alpha=1.0):
    text_v_offset = -0.01

    # data space coordinates to find new point after rotated
    fixed_point_rotation = Affine2D().rotate_deg_around(box_x, box_y, angle)

    # add rectangle at corner position and rotate by angle
    rect = patches.Rectangle((box_x, box_y), box_width, box_height, angle=angle, linewidth=linewidth,
                             edgecolor=edgecolor,
                             facecolor=facecolor, clip_on=False, alpha=alpha)
    ax.add_patch(rect)

    # put angle within +180/-180
    normalized_angle = angle
    if angle > 0:
        while normalized_angle > 180:
            normalized_angle -= 360
    elif angle < 0:
        while normalized_angle < -180:
            normalized_angle += 360

    # angle the textbox nicely
    if aligned_text:
        text_angle = normalized_angle
        if abs(text_angle) > 90:
            text_angle += 180
    else:
        text_angle = 0

    # space the text box nicely so it fits no matter orientation
    # nice centering depends on orientation of text in the figure
    if aligned_text:

        # upper quadrants and bottom quadrants have different text orientation and adjustment
        if abs(normalized_angle) > 90:
            # upper quadrant adjustment
            rect_center_pos = [box_x + box_width / 2, box_y + box_height * 0.5 - text_v_offset]
        else:
            # bottom quadrant adjustment
            rect_center_pos = [box_x + box_width / 2, box_y + box_height * 0.5 + text_v_offset]
    else:
        rect_center_pos = [box_x + box_width / 2, box_y + box_height * 0.5]

    # rotate around rectangle corner
    text_pos = fixed_point_rotation.transform(rect_center_pos)

    # unaligned text uses standard vertical offset in axes frame
    if not aligned_text:
        text_pos[1] += text_v_offset

    # add text box to center of rectangle
    ax.text(text_pos[0], text_pos[1], text_str, rotation=text_angle, rotation_mode='anchor',
            fontsize=fontsize, va='center', ha='center', clip_on=False, alpha=alpha)
    # , bbox={'facecolor':'none', 'edgecolor':'k'})


# multiple period grids per angle
def plot_2D_hypergrid_similarity_1(n=100):
    results = glob.glob("out/hypergrid_2D_similarity_frames_1_*")
    for filename in results:
        os.remove(filename)
    filename_str = "out/hypergrid_2D_similarity_frames_1_%05u.png"

    sns.set()

    # trajectory for moving point
    xs, ys = datasets.lemiscate(n=n, alpha=1.2)
    ys = 3 * ys

    num_points = len(xs)
    fig = plt.figure(num=1, figsize=(12, 8))  # , constrained_layout=True)

    n_bins = 4
    n_subspace_dims = 1
    n_input_dims = 2
    n_angles = 8
    n_grids_per_angle = 3
    n_grids = n_angles * n_grids_per_angle

    all_periods = []
    # dist_periods = [0.5, 1.0, 1.5]
    uniform_periods = list(np.linspace(0, 2, endpoint=False, num=(n_grids_per_angle + 1))[1:])
    # print(uniform_periods)

    for k in range(n_angles):
        all_periods += uniform_periods
    # print(all_periods)
    # dist_periods = [0.5 for k in range(n_grids)]

    gridTransform = create_transform_2D_to_1D_hypergrids(n_input_dims=n_input_dims, n_grids=n_grids, n_bins=n_bins,
                                                         n_subspace_dims=n_subspace_dims, periods=all_periods)

    # evaluation of the space
    # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    h = 0.04
    xx, yy = np.meshgrid(np.arange(x_min, x_max + h, h),
                         np.arange(y_min, y_max + h, h))
    print("mesh:")
    print(xx.shape)
    print(yy.shape)

    # mesh put together with x/y coordinates
    XY_mesh = np.c_[xx.ravel(), yy.ravel()]
    print(XY_mesh.shape)

    XY_gnomes = gridTransform.transform(XY_mesh)
    print(XY_gnomes.shape)
    XY_gnomes = XY_gnomes.reshape(xx.shape[0], xx.shape[1], -1)
    print(XY_gnomes.shape)

    # Z = clf.predict_proba(X_mesh)
    # CS = ax.contour(X, Y, Z)

    # Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # deep_cmap = ListedColormap(sns.color_palette("deep"))
    # colors = sns.husl_palette(8, s=.45)
    colors = sns.color_palette("deep", n_colors=n_angles)

    for point_index in range(num_points):

        # gs = gridspec.GridSpec(1, 2, height_ratios=[1, 0.5])
        # ax0 = plt.subplot(gs[0])
        # ax1 = plt.subplot(gs[1])

        fig, axes = plt.subplots(1, 2, num=1, gridspec_kw={"width_ratios": [1, 0.4]})

        # axes for hypergrid visual
        ax0 = axes[0]

        fig.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=1)
        # fig.subplots_adjust(wspace=0.8)
        random.seed(6)

        points = [(xs[point_index], ys[point_index])]
        X_gnomes = gridTransform.transform(points)
        mesh_gnome_result = gnome_similarity(X_gnomes.reshape(len(points), -1),
                                                                 XY_gnomes.reshape(-1, XY_gnomes.shape[2]))
        mesh_gnome_result = mesh_gnome_result.reshape(XY_gnomes.shape[0:2])
        ax0.contourf(xx, yy, mesh_gnome_result, levels=10)  # , colors=newcolors)

        ax0.set_aspect('equal')
        ax0.set_xlim(-2, 2)
        ax0.set_ylim(-2, 2)
        ax0.tick_params(labelsize=8)
        tick_locs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        ax0.xaxis.set_major_locator(ticker.FixedLocator(tick_locs))
        ax0.yaxis.set_major_locator(ticker.FixedLocator(tick_locs))

        # palettes
        # --------
        # deep, muted, bright, pastel, dark, colorblind

        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        # colors = sns.color_palette("deep")  # , n_colors=3)

        for i, pnt in enumerate(points):
            ax0.scatter(pnt[0], pnt[1], color='k', marker='o', s=30)

        unit_periods = [1.0]
        box_height = 0.2
        dist_periods = [0.5, 1.0, 1.5]

        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        # plot_proj_dists = [2.3, 3.2, 2.3, 3.2, 2.3, 3.2, 2.3, 3.2]

        plot_proj_dists = [2.6, 3.2]

        for k, angle in enumerate(angles):

            plot_dist = plot_proj_dists[k % 2]

            # grid_colors = [colors[k] for k in range(len(dist_periods))]
            grid_colors = [colors[k] for j in range(len(dist_periods))]

            if k % 2 == 1:
                aligned_text = True
            else:
                aligned_text = False

            add_1D_hypergrid(ax0, (0, 0), plot_dist, angle=angles[k], points=points, periods=dist_periods,
                             colors=grid_colors,
                             box_height=box_height, aligned_text=aligned_text)

        # plot fading line
        # path = mpath.Path(np.column_stack([xs, ys]))
        # verts = path.interpolated(steps=3).vertices
        # x, y = verts[:, 0], verts[:, 1]
        # z = np.linspace(0, 1, len(x))
        # lc = colorline(x, y, z, ax=ax0, cmap=plt.get_cmap('jet'), linewidth=2)

        # axes for gnome visual
        ax1 = axes[1]

        ref_points = [(0, 0), (1, 1)]
        ref_gnomes = gridTransform.transform(ref_points)
        ref_gnomes[:, 0:3] = True

        otsuka_result = hypergrid_transform.otsuka_similarity(X_gnomes.reshape(len(points), -1),
                                                              ref_gnomes.reshape(len(ref_points), -1))
        tanimoto_result = hypergrid_transform.tanimoto_similarity(X_gnomes.reshape(len(points), -1),
                                                                  ref_gnomes.reshape(len(ref_points), -1))
        gnome_result = gnome_similarity(X_gnomes.reshape(len(points), -1),
                                                            ref_gnomes.reshape(len(ref_points), -1))
        weighted_result = hypergrid_transform.weighted_similarity(X_gnomes.reshape(len(points), -1),
                                                                  ref_gnomes.reshape(len(ref_points), -1))

        print("points:", points)
        print("Similarities")
        print("gnome, tanimoto, otsuka, weighted")
        print(gnome_result, tanimoto_result, otsuka_result, weighted_result)
        print(gnome_result.shape)
        print()

        # remove excess axis for samples
        X_gnomes = X_gnomes.reshape(n_grids, n_bins).astype(int)

        indices = np.where(X_gnomes == 1)
        X_gnomes[indices] = indices[0] + 1
        # for k in range(n_grids):
        #    X_gnomes[k]
        #    X_gnomes[X_gnomes==1][k,:] = k+1
        # X_gnomes[k,X_gnomes==1] = k+1

        # rotated gnome state data
        # rotated_gnome_data = np.rot90(X_gnomes, k=-1, axes=(1, 0))

        # periods labels
        # subspace_periods = gridTransform.subspace_periods
        # periods = list(subspace_periods.reshape(-1))
        # periods = ["%.2f" % period for period in periods]

        # plot grid as a binary heatmap
        # cmap = ListedColormap(['white', 'gray'])
        # colors2 = ['white'] + colors
        # colordict = {0: np.array(colorConverter.to_rgba("white"))}
        colors2 = [np.array(colorConverter.to_rgba("white"))]
        for k in range(n_grids):
            curr_color = np.array(colorConverter.to_rgba(colors[k // n_grids_per_angle]))
            colors2.append(curr_color)

        cmap = ListedColormap(colors2)
        annot = np.zeros(shape=X_gnomes.shape)
        for k in range(n_bins):
            annot[:, k] = k

        gnome_plot = sns.heatmap(X_gnomes, ax=ax1, square=True, linewidths=0.05, linecolor='k', cbar=False,
                                 cmap=colors2, annot=annot, annot_kws={'color': 'k', 'alpha': 0.4}, clip_on=False)

        # X-Axis bin labels
        gnome_plot.set_xticklabels([])
        # gnome_plot.set_xticklabels([k for k in range(n_bins)], fontdict={'fontsize': 12})

        # Y-Axis hypergrid angle labels
        text_angles = []
        for angle in angles:
            text_angles.append("%.0f$^\circ$" % angle)

        gnome_plot.yaxis.set_major_locator(ticker.IndexLocator(n_grids_per_angle, n_grids_per_angle / 2))
        gnome_plot.set_yticklabels(text_angles, rotation=0, fontdict={'fontsize': 12})
        gnome_plot.set_aspect('equal')

        plt.savefig(filename_str % point_index)  # , bbox_inches='tight')
        plt.clf()

        # break

    plt.close()

    sns.reset_defaults()


# multiple period grids per angle
def plot_2D_hypergrid_similarity_2(n=100, test_config=None,
                                   filename_str="out/hypergrid_2D_similarity_frames_2_%05u.png"):
    results = glob.glob("out/hypergrid_2D_similarity_frames_2_*")
    for filename in results:
        os.remove(filename)
    # filename_str = "out/hypergrid_2D_similarity_frames_2_%05u.png"

    sns.set()

    # trajectory for moving point
    xs, ys = datasets.lemiscate(n=n, alpha=1.2)
    ys = 3 * ys

    num_points = len(xs)
    fig = plt.figure(num=1, figsize=(12, 8))  # , constrained_layout=True)

    # angles = [0, 45, 90, 135]
    # angles = [0, 90]

    # angles = []
    # for k in range(n_angles):
    #   angle = 360 * k / n_angles
    #   angles.append(angle)

    # test_config['periods'] = list(np.linspace(0, 2, endpoint=False, num=(n_grids_per_angle + 1))[1:])

    # 4 projections example
    if test_config is None:
        test_config = dict(n_bins=4,
                           n_subspace_dims=1,
                           n_input_dims=2,
                           n_angles=4,
                           n_grids_per_angle=1,
                           angles=[0, 45, 90, 135],
                           periods=[1.0, ])

    print("config:")
    print(test_config)

    # unit_periods = [1.0]
    box_height = 0.2
    plot_proj_dists = [2.6, 3.2]

    n_bins = test_config['n_bins']
    n_subspace_dims = test_config['n_subspace_dims']
    n_input_dims = test_config['n_input_dims']
    n_angles = test_config['n_angles']
    n_grids_per_angle = test_config['n_grids_per_angle']
    angles = test_config['angles']
    per_projection_periods = test_config['periods']

    n_grids = n_angles * n_grids_per_angle

    all_periods = []
    # dist_periods = [0.5, 1.0, 1.5]
    # uniform_periods = list(np.linspace(1, 3, endpoint=False, num=(n_grids_per_angle + 1))[1:])
    # print(uniform_periods)

    for k in range(n_angles):
        all_periods += per_projection_periods

    basis_angles = []
    for i in range(len(angles)):
        for j in range(n_grids_per_angle):
            basis_angles.append(angles[i])

    # angles = [0, 45, 90, 135, 180, 225, 270, 315]
    # print(all_periods)
    # dist_periods = [0.5 for k in range(n_grids)]

    print("box_height =", box_height)

    print("n_input_dims =", n_input_dims)
    print("n_grids_per_angle =", n_grids_per_angle)
    print("n_bins =", n_bins)
    print("n_subspace_dims =", n_subspace_dims)
    print("n_grids =", n_grids)

    print("basis angles:")
    print(basis_angles)

    print("all_periods:")
    print(all_periods)

    gridTransform = create_transform_2D_to_1D_hypergrids(n_input_dims=n_input_dims,
                                                         n_grids=n_grids, n_bins=n_bins,
                                                         n_subspace_dims=n_subspace_dims, periods=all_periods,
                                                         angles=basis_angles)

    print("dist_periods:")
    print(per_projection_periods)

    # evaluation of the space
    # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    h = 0.04
    xx, yy = np.meshgrid(np.arange(x_min, x_max + h, h),
                         np.arange(y_min, y_max + h, h))
    print("mesh x/y size:", xx.shape, yy.shape)

    # mesh put together with x/y coordinates
    XY_mesh = np.c_[xx.ravel(), yy.ravel()]
    print("evaluation meshgrid shape")
    print(XY_mesh.shape)

    XY_gnomes = gridTransform.transform(XY_mesh)
    # print(XY_gnomes.shape)
    XY_gnomes = XY_gnomes.reshape(xx.shape[0], xx.shape[1], -1)
    print("evaluation result shape")
    print(XY_gnomes.shape)

    # Z = clf.predict_proba(X_mesh)
    # CS = ax.contour(X, Y, Z)

    # Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # heat_colors = sns.light_palette("red", n_colors=10)
    heat_colors = sns.color_palette("Reds", n_colors=10)
    # heat_colors.reverse()
    # print(heat_colors)
    for i, clr in enumerate(heat_colors):
        temp_color = list(heat_colors[i])
        if len(temp_color) == 3:
            temp_color += [1.0, ]
        temp_color[3] = i / 10
        heat_colors[i] = temp_color

    print("similarity contour colors")
    print(heat_colors)
    no_norm = NoNorm(vmin=0, vmax=1.0, clip=True)
    heat_cmap = ListedColormap(heat_colors)

    """
    red_color = [0.579361783929258, 0.04244521337946944, 0.07361783929257976, 1.0]
    heat_colors = []
    for i in range(10):
        temp_color = deepcopy(red_color)
        temp_color[3] = i/10
        heat_colors.append(temp_color)

    print(heat_colors)
    """

    # deep_cmap = ListedColormap(sns.color_palette("deep"))
    # colors = sns.husl_palette(8, s=.45)
    colors = sns.color_palette("deep", n_colors=n_angles)

    for point_index in range(num_points):

        # gs = gridspec.GridSpec(1, 2, height_ratios=[1, 0.5])
        # ax0 = plt.subplot(gs[0])
        # ax1 = plt.subplot(gs[1])

        fig, axes = plt.subplots(1, 2, num=1, gridspec_kw={"width_ratios": [1, 0.4]})

        # axes for hypergrid visual
        ax0 = axes[0]

        # 4 projections <= 135 subplot adjustment
        fig.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=0.5)

        # 8 projections subplot adjustment
        # fig.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=1)

        random.seed(6)

        points = [(xs[point_index], ys[point_index])]
        X_gnomes = gridTransform.transform(points)
        mesh_gnome_result = gnome_similarity(X_gnomes.reshape(len(points), -1),
                                                                 XY_gnomes.reshape(-1, XY_gnomes.shape[2]))
        mesh_gnome_result = mesh_gnome_result.reshape(XY_gnomes.shape[0:2])

        print(point_index, mesh_gnome_result.max())

        # ax0.contourf(xx, yy, mesh_gnome_result, vmin=0, vmax=1.0, levels=10, norm=no_norm, colors=heat_colors)  # , colors=newcolors)
        # ax0.contourf(xx, yy, mesh_gnome_result, vmin=0, vmax=1.0, levels=10, norm=no_norm, cmap=heat_cmap)  # , colors=newcolors)

        ax0.contourf(xx, yy, mesh_gnome_result, levels=10, norm=no_norm, cmap=heat_cmap)

        ax0.set_aspect('equal')
        ax0.set_xlim(-2, 2)
        ax0.set_ylim(-2, 2)
        ax0.tick_params(labelsize=8)
        tick_locs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        ax0.xaxis.set_major_locator(ticker.FixedLocator(tick_locs))
        ax0.yaxis.set_major_locator(ticker.FixedLocator(tick_locs))

        # desc_str_0 = "Point Projection to Hypergrids\n&\nHeatmap of Similarity Measure"
        desc_str_0 = "Mapping to 1D Hypergrids / Similarity Measure"
        ax0.set_title(desc_str_0)
        # ax0.text(0.5, -0.2, desc_str_0, size=12, ha="center", transform=ax0.transAxes)

        # palettes
        # --------
        # deep, muted, bright, pastel, dark, colorblind

        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        # colors = sns.color_palette("deep")  # , n_colors=3)

        for i, pnt in enumerate(points):
            ax0.scatter(pnt[0], pnt[1], color='white', edgecolors='k', linewidth=1, marker='o', s=30, zorder=10)

        # dist_periods = [0.5, 1.0, 1.5]
        dist_periods = list(per_projection_periods)

        # plot_proj_dists = [2.3, 3.2, 2.3, 3.2, 2.3, 3.2, 2.3, 3.2]

        for k, angle in enumerate(angles):

            plot_dist = plot_proj_dists[k % 2]

            # grid_colors = [colors[k] for k in range(len(dist_periods))]
            grid_colors = [colors[k] for j in range(len(dist_periods))]

            if k % 2 == 1:
                aligned_text = True
            else:
                aligned_text = False

            add_1D_hypergrid(ax0, (0, 0), plot_dist, angle=angles[k], points=points, periods=dist_periods,
                             n_bins=n_bins, colors=grid_colors,
                             box_height=box_height, aligned_text=aligned_text)

        # plot fading line
        # path = mpath.Path(np.column_stack([xs, ys]))
        # verts = path.interpolated(steps=3).vertices
        # x, y = verts[:, 0], verts[:, 1]
        # z = np.linspace(0, 1, len(x))
        # lc = colorline(x, y, z, ax=ax0, cmap=plt.get_cmap('jet'), linewidth=2)

        ax1 = axes[1]

        ref_points = [(0, 0), (1, 1)]
        ref_gnomes = gridTransform.transform(ref_points)
        # ref_gnomes[:, 0:3] = True

        otsuka_result = hypergrid_transform.otsuka_similarity(X_gnomes.reshape(len(points), -1),
                                                              ref_gnomes.reshape(len(ref_points), -1))
        tanimoto_result = hypergrid_transform.tanimoto_similarity(X_gnomes.reshape(len(points), -1),
                                                                  ref_gnomes.reshape(len(ref_points), -1))
        gnome_result = gnome_similarity(X_gnomes.reshape(len(points), -1),
                                                            ref_gnomes.reshape(len(ref_points), -1))
        weighted_result = hypergrid_transform.weighted_similarity(X_gnomes.reshape(len(points), -1),
                                                                  ref_gnomes.reshape(len(ref_points), -1))

        """
        print("points:", points)
        print("Similarities")
        print("gnome, tanimoto, otsuka, weighted")
        print(gnome_result, tanimoto_result, otsuka_result, weighted_result)
        print(gnome_result.shape)
        print()
        """

        # remove excess axis for samples
        X_gnomes = X_gnomes.reshape(n_grids, n_bins).astype(int)

        indices = np.where(X_gnomes == 1)
        X_gnomes[indices] = indices[0] + 1
        # for k in range(n_grids):
        #    X_gnomes[k]
        #    X_gnomes[X_gnomes==1][k,:] = k+1
        # X_gnomes[k,X_gnomes==1] = k+1

        # rotated gnome state data
        # rotated_gnome_data = np.rot90(X_gnomes, k=-1, axes=(1, 0))

        # periods labels
        # subspace_periods = gridTransform.subspace_periods
        # periods = list(subspace_periods.reshape(-1))
        # periods = ["%.2f" % period for period in periods]

        # plot grid as a binary heatmap
        # cmap = ListedColormap(['white', 'gray'])
        # colors2 = ['white'] + colors
        # colordict = {0: np.array(colorConverter.to_rgba("white"))}
        colors2 = [np.array(colorConverter.to_rgba("white"))]
        for k in range(n_grids):
            curr_color = np.array(colorConverter.to_rgba(colors[k // n_grids_per_angle]))
            colors2.append(curr_color)

        cmap = ListedColormap(colors2)
        annot = np.zeros(shape=X_gnomes.shape)
        for k in range(n_bins):
            annot[:, k] = k

        gnome_plot = sns.heatmap(X_gnomes, ax=ax1, square=True, linewidths=0.05, linecolor='k', cbar=False,
                                 cmap=colors2, annot=annot, annot_kws={'color': 'k', 'alpha': 0.4}, clip_on=False)

        # X-Axis bin labels
        gnome_plot.set_xticklabels([])
        # gnome_plot.set_xticklabels([k for k in range(n_bins)], fontdict={'fontsize': 12})

        # Y-Axis hypergrid angle labels
        text_angles = []
        for angle in angles:
            text_angles.append("%.0f$^\circ$" % angle)

        #gnome_plot.yaxis.set_major_locator(ticker.IndexLocator(n_grids_per_angle, n_grids_per_angle / 2))
        gnome_plot.yaxis.set_major_locator(ticker.NullLocator())
        #gnome_plot.set_yticklabels(text_angles, rotation=0, fontdict={'fontsize': 12})
        gnome_plot.set_aspect('equal')

        title_str_1 = "Distributed Binary Representation\n"
        #title_str_1 += "active/total = %d/%d\n" % (n_grids, n_grids * n_bins)
        #title_str_1 += "w=%d k=%d\n" % (n_grids, n_grids * n_bins)
        # title_str_1 += "total bits = %d\n" % (n_grids*n_bins,)
        gnome_plot.set_title(title_str_1)

        # format_str = r'{\fontsize{30pt}{3em}\selectfont{}{Mean WRFv3.5 LHF\r}' \
        #             r'{\fontsize{18pt}{3em}\selectfont{}' \
        #             r'(September 16 - October 30, 2012)}'
        desc_str_1 = r'{\fontsize{30pt}{3em}\selectfont{}{Parameters}\n' \
                     r'{\fontsize{18pt}{3em}\selectfont{}' \
                     r'(input dimension)\n}' \
                     r'(projection angles)\n}' \
                     r'(bins per grid)\n}' \
                     r'(grid periods)\n}' \
                     r'(representation k bits / (n_grids*n_bins) bits)}'

        desc_str_1 = "Hypergrid Transform Parameters:\n\n"
        desc_str_1 += "input dimension n=%d\n" % n_input_dims
        desc_str_1 += "subspace dimension m=%d\n" % 1
        #desc_str_1 += " = %d\n" % n_angles
        desc_str_1 += "bin frequency k=%d\n" % n_bins
        #desc_str_1 += "p = %d" % n_grids_per_angle
        desc_str_1 += "grid period p=%.1f" % per_projection_periods[0]
        # desc_str_1 += "active bits = %d\n" % n_grids
        # desc_str_1 += "total bits = %d\n" % (n_grids*n_bins,)

        # box color
        sns.color_palette("deep")

        # plt.rc('text', usetex=True)

        # gnome_plot.text(n_bins/2, n_grids, "(a) my label", size=12, ha="center", transform=gnome_plot.transData)
        # bbox=dict(facecolor=sns.color_palette("deep")[5], alpha=0.3),
        gnome_plot.text(n_bins + 2, n_grids + 2, desc_str_1, size=12, ha="right", va="top",
                        bbox=dict(facecolor="wheat", alpha=0.5, boxstyle='round', edgecolor='k'),
                        transform=gnome_plot.transData)

        plt.savefig(filename_str % point_index)  # , bbox_inches='tight')
        plt.clf()

        # break

    plt.close()

    sns.reset_defaults()


# multiple period grids per angle
def plot_2D_hypergrid_similarity_figures(n=100, test_config=None,
                                   filename_str="out/hypergrid_2D_similarity_frames_2_%05u.png"):

    sns.set()

    # trajectory for moving point
    xs, ys = datasets.lemiscate(n=n, alpha=1.2)
    ys = 3 * ys

    num_points = len(xs)
    fig = plt.figure(num=1, figsize=(12, 8))  # , constrained_layout=True)

    # 4 projections example
    if test_config is None:
        test_config = dict(n_bins=4,
                           n_subspace_dims=1,
                           n_input_dims=2,
                           n_angles=4,
                           n_grids_per_angle=1,
                           angles=[0, 45, 90, 135],
                           periods=[1.0, ])

    print("config:")
    print(test_config)

    # unit_periods = [1.0]
    box_height = 0.2
    plot_proj_dists = [2.6, 3.2]

    n_bins = test_config['n_bins']
    n_subspace_dims = test_config['n_subspace_dims']
    n_input_dims = test_config['n_input_dims']
    n_angles = test_config['n_angles']
    n_grids_per_angle = test_config['n_grids_per_angle']
    angles = test_config['angles']
    per_projection_periods = test_config['periods']

    n_grids = n_angles * n_grids_per_angle

    all_periods = []

    for k in range(n_angles):
        all_periods += per_projection_periods

    basis_angles = []
    for i in range(len(angles)):
        for j in range(n_grids_per_angle):
            basis_angles.append(angles[i])

    gridTransform = create_transform_2D_to_1D_hypergrids(n_input_dims=n_input_dims,
                                                         n_grids=n_grids, n_bins=n_bins,
                                                         n_subspace_dims=n_subspace_dims, periods=all_periods,
                                                         angles=basis_angles)

    # evaluation of the space
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    h = 0.04
    xx, yy = np.meshgrid(np.arange(x_min, x_max + h, h),
                         np.arange(y_min, y_max + h, h))

    # mesh put together with x/y coordinates
    XY_mesh = np.c_[xx.ravel(), yy.ravel()]

    XY_gnomes = gridTransform.transform(XY_mesh)
    XY_gnomes = XY_gnomes.reshape(xx.shape[0], xx.shape[1], -1)

    heat_colors = sns.color_palette("Reds", n_colors=10)
    for i, clr in enumerate(heat_colors):
        temp_color = list(heat_colors[i])
        if len(temp_color) == 3:
            temp_color += [1.0, ]
        temp_color[3] = i / 10
        heat_colors[i] = temp_color

    no_norm = NoNorm(vmin=0, vmax=1.0, clip=True)
    heat_cmap = ListedColormap(heat_colors)

    colors = sns.color_palette("deep", n_colors=n_angles)

    for point_index in range(num_points):

        #fig, axes = plt.subplots(1, 2, num=1, gridspec_kw={"width_ratios": [1, 0.4]})
        fig, ax0 = plt.subplots(1, 1, num=1)

        # 4 projections <= 135 subplot adjustment
        #fig.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=0.5)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.8)#, wspace=0.5)

        random.seed(6)

        points = [(xs[point_index], ys[point_index])]
        X_gnomes = gridTransform.transform(points)
        mesh_gnome_result = gnome_similarity(X_gnomes.reshape(len(points), -1),
                                                                 XY_gnomes.reshape(-1, XY_gnomes.shape[2]))
        mesh_gnome_result = mesh_gnome_result.reshape(XY_gnomes.shape[0:2])

        ax0.contourf(xx, yy, mesh_gnome_result, levels=10, norm=no_norm, cmap=heat_cmap)

        ax0.set_aspect('equal')
        ax0.set_xlim(-2, 2)
        ax0.set_ylim(-2, 2)
        ax0.tick_params(labelsize=8)
        tick_locs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        ax0.xaxis.set_major_locator(ticker.FixedLocator(tick_locs))
        ax0.yaxis.set_major_locator(ticker.FixedLocator(tick_locs))

        for i, pnt in enumerate(points):
            ax0.scatter(pnt[0], pnt[1], color='white', edgecolors='k', linewidth=1, marker='o', s=30, zorder=10)

        dist_periods = list(per_projection_periods)

        for k, angle in enumerate(angles):

            plot_dist = plot_proj_dists[k % 2]

            # grid_colors = [colors[k] for k in range(len(dist_periods))]
            grid_colors = [colors[k] for j in range(len(dist_periods))]

            if k % 2 == 1:
                aligned_text = True
            else:
                aligned_text = False

            add_1D_hypergrid(ax0, (0, 0), plot_dist, angle=angles[k], points=points, periods=dist_periods,
                             n_bins=n_bins, colors=grid_colors,
                             box_height=box_height, aligned_text=aligned_text)

        ref_points = [(0, 0), (1, 1)]
        ref_gnomes = gridTransform.transform(ref_points)

        # remove excess axis for samples
        X_gnomes = X_gnomes.reshape(n_grids, n_bins).astype(int)

        indices = np.where(X_gnomes == 1)
        X_gnomes[indices] = indices[0] + 1

        colors2 = [np.array(colorConverter.to_rgba("white"))]
        for k in range(n_grids):
            curr_color = np.array(colorConverter.to_rgba(colors[k // n_grids_per_angle]))
            colors2.append(curr_color)

        cmap = ListedColormap(colors2)
        annot = np.zeros(shape=X_gnomes.shape)
        for k in range(n_bins):
            annot[:, k] = k

        #gnome_plot = sns.heatmap(X_gnomes, ax=ax1, square=True, linewidths=0.05, linecolor='k', cbar=False,
        #                         cmap=colors2, annot=annot, annot_kws={'color': 'k', 'alpha': 0.4}, clip_on=False)

        desc_str_1 = "Hypergrid Transform Parameters:\n\n"
        desc_str_1 += "input dimension n=%d\n" % n_input_dims
        desc_str_1 += "subspace dimension m=%d\n" % 1
        desc_str_1 += "bin frequency k=%d\n" % n_bins
        desc_str_1 += "grid period p=%.1f" % per_projection_periods[0]

        # box color
        sns.color_palette("deep")

        # plt.rc('text', usetex=True)

        plt.savefig(filename_str % point_index)  # , bbox_inches='tight')
        plt.clf()

    plt.close()

    sns.reset_defaults()



if __name__ == "__main__":
    # plot_2D_hypergrid_similarity_1(n=200)
    # plot_2D_hypergrid_similarity_2(n=200)
    test_count = 0

    #for n_angles in [2, 4]:
    for n_angles in [1, 2]:
        #if n_angles == 2:
        if n_angles == 1:
            angles = [0]
        #elif n_angles == 4:
        elif n_angles == 2:
            angles = [0, 90]
        else:
            print("n_angles =", n_angles)
            raise Exception()

        for n_bins in [4, 8]:
            #for n_grids_per_angle in [1, 2, 3]:
            for n_grids_per_angle in [1]:

                if n_grids_per_angle == 1:
                    periods = [1.0, ]
                    # periods = list(np.linspace(0, 2, endpoint=False, num=(n_grids_per_angle + 1))[1:])
                elif n_grids_per_angle == 2:
                    # periods = [0.67,1.33]
                    periods = list(np.linspace(0, 2, endpoint=False, num=(n_grids_per_angle + 1))[1:])
                elif n_grids_per_angle == 3:
                    periods = list(np.linspace(0, 2, endpoint=False, num=(n_grids_per_angle + 1))[1:])
                else:
                    periods = list(np.linspace(0, 2, endpoint=False, num=(n_grids_per_angle + 1))[1:])

                test_config = dict(n_bins=n_bins,
                                   n_subspace_dims=1,
                                   n_input_dims=2,
                                   n_angles=n_angles,
                                   n_grids_per_angle=n_grids_per_angle,
                                   angles=angles,
                                   periods=periods)

                #print("test_config:")
                #print(test_config)
                test_count += 1
                filename_str = "grid_2D_%d_angles_%d_periods_%d_bins_frames" % (n_angles, n_grids_per_angle, n_bins)
                filename_str += "_%05u.png"
                print(filename_str)

                #plot_2D_hypergrid_similarity_2(n=1, test_config=test_config, filename_str=filename_str)
                plot_2D_hypergrid_similarity_figures(n=1, test_config=test_config, filename_str=filename_str)

    print("total tests:", test_count)
    """
    # 4 projections in 45 deg increments, 8 bins
    test_config = dict(n_bins=8,
                       n_subspace_dims=1,
                       n_input_dims=2,
                       n_angles=4,
                       n_grids_per_angle=1,
                       angles=[0, 45, 90, 135],
                       periods = [1.0,])

    plot_2D_hypergrid_similarity_2(n=200, test_config=test_config,
                                   filename_str="out/grid_2D_0_45_90_135_angles_1.0_periods_8_bins_v1_frames_%05u.png")
    """

    """
    # 4 projections in 45 deg increments, 4 bins
    test_config = dict(n_bins=4,
                       n_subspace_dims=1,
                       n_input_dims=2,
                       n_angles=4,
                       n_grids_per_angle=1,
                       angles=[0, 45, 90, 135],
                       periods = [1.0,])
                       
    plot_2D_hypergrid_similarity_2(n=200, test_config=test_config,
                                   filename_str="out/grid_2D_0_45_90_135_angles_1.0_periods_4_bins_v1_frames_%05u.png")
    """

    """
    # 2 orthogonal projections, 4 bins
    test_config = dict(n_bins=4,
                       n_subspace_dims=1,
                       n_input_dims=2,
                       n_angles=2,
                       n_grids_per_angle=1,
                       angles=[0, 90],
                       periods=[1.0, ])

    plot_2D_hypergrid_similarity_2(n=200, test_config=test_config,
                                   filename_str="out/grid_2D_0_90_angles_1.0_periods_4_bins_v1_frames_%05u.png")

    # 2 orthogonal projections, 8 bins
    test_config = dict(n_bins=8,
                       n_subspace_dims=1,
                       n_input_dims=2,
                       n_angles=2,
                       n_grids_per_angle=1,
                       angles=[0, 90],
                       periods=[1.0, ])

    plot_2D_hypergrid_similarity_2(n=200, test_config=test_config,
                                   filename_str="out/grid_2D_0_90_angles_1.0_periods_8_bins_v1_frames_%05u.png")
    """
