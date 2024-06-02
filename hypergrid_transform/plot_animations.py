import numpy as np


import os
import glob

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, colorConverter
import matplotlib.patches as patches
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import random

from matplotlib import ticker
from matplotlib.transforms import Affine2D
from brainblocks.datasets import time_series as datasets
# import helpers.hypergrid_graphics as hg

#import hypergrid_transform
#from hypergrid_transform import HyperGridTransform
from brainblocks.tools import HyperGridTransform
from brainblocks.metrics import *
import seaborn as sns


def create_transform_2D_to_1D_hypergrids(n_input_dims=2, n_subspace_dims=1, n_grids=1, n_bins=4,
                                         periods=None):
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
    if n_input_dims == 2:
        for k in range(n_grids):
            angle = 2 * np.pi * k / n_grids
            custom_bases[k][0] = [np.cos(angle), np.sin(angle)]

    custom_periods = np.ones((n_grids, n_subspace_dims))
    if not periods is None:
        for k in range(n_grids):
            for j in range(n_subspace_dims):
                custom_periods[k, j] = periods[k * n_subspace_dims + j]

    # print(custom_periods)

    configs = {"n_input_dims": n_input_dims, "n_grids": n_grids, "n_bins": n_bins, "n_acts": 1,
               "n_subspace_dims": n_subspace_dims,
               "flatten_output": False, "max_period": 2.0, "min_period": 0.05,
               "use_normal_dist_bases": False,
               "use_standard_bases": False,
               "use_orthogonal_bases": False,
               "set_bases": custom_bases,
               # "use_random_uniform_periods": True,
               "set_periods": custom_periods
               }
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

        ax.plot([pnt[0], line_x], [pnt[1], line_y], clip_on=False, color=colors[iter_index], alpha=0.3)  # , zorder=6)

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
    label_padding = 3 * box_height
    label_x, label_y = projection_point_rotation.transform(
        [rotation_point[0] + projection_length + n_grids * box_height + label_padding, rotation_point[1]])

    ax.text(label_x, label_y, "%.0f$^\circ$" % pos_angle, rotation=transform_angle, rotation_mode='anchor',
            fontsize=16, va='center', ha='center', clip_on=False)


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


def plot_2D_hypergrid_frames_1(n=100):
    filename_str = "out/hypergrid_2D_frames_1_%05u.png"

    sns.set()
    # xs, ys = datasets.spiral(100)
    # xs, ys = datasets.circle(n=100, r=1.5)
    xs, ys = datasets.lemiscate(n=n, alpha=1.2)
    ys = 3 * ys

    num_points = len(xs)
    fig = plt.figure(num=1, figsize=(10, 8))  # , constrained_layout=True)

    for point_index in range(num_points):

        fig, ax = plt.subplots(1, 1, num=1)
        fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
        random.seed(6)
        ax.set_aspect('equal')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        colors = sns.color_palette("colorblind")  # , n_colors=3)
        colors = sns.color_palette("pastel")  # , n_colors=3)

        points = [(xs[point_index], ys[point_index])]
        for i, pnt in enumerate(points):
            ax.scatter(pnt[0], pnt[1], color=colors[i])

        periods1 = [0.7, 1.35]
        periods2 = [0.7, 1.35, 1.7]
        periods3 = [0.3, 0.7, 1.35]
        periods4 = [0.3, 0.5, 0.8, 1.1]

        add_1D_hypergrid(ax, (0, 0), 2.3, angle=0, points=points, periods=periods1, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 3.2, angle=45, points=points, periods=periods1, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 2.3, angle=90, points=points, periods=periods2, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 3.2, angle=135, points=points, periods=periods2, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 2.3, angle=180, points=points, periods=periods3, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 3.2, angle=225, points=points, periods=periods3, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 2.3, angle=270, points=points, periods=periods4, aligned_text=True, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 3.2, angle=315, points=points, periods=periods4, aligned_text=True, colors=colors)

        plt.savefig(filename_str % point_index)  # , bbox_inches='tight')
        plt.clf()

    plt.close()

    # Generate a blue-white-red palette:
    color_list = sns.diverging_palette(240, 10, n=9)

    # Generate a brighter green-white-purple palette
    # color_list = sns.diverging_palette(150, 275, s=80, l=55, n=9)

    # Generate a blue-black-red palette
    # color_list = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark")

    # Generate a colormap object
    # cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

    sns.reset_defaults()


# multi-point plot
def plot_2D_hypergrid_frames_2(n=100):
    results = glob.glob("out/hypergrid_2D_frames_1_*")
    for filename in results:
        os.remove(filename)
    filename_str = "out/hypergrid_2D_frames_1_%05u.png"

    sns.set()
    # xs, ys = datasets.spiral(100)
    # xs, ys = datasets.circle(n=100, r=1.5)
    xs, ys = datasets.lemiscate(n=n, alpha=1.2)
    ys = 3 * ys

    num_points = len(xs)
    fig = plt.figure(num=1, figsize=(10, 8))  # , constrained_layout=True)

    for point_index in range(num_points):

        fig, ax = plt.subplots(1, 1, num=1)
        fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
        random.seed(6)
        ax.set_aspect('equal')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        # palettes
        # --------
        # deep, muted, bright, pastel, dark, colorblind

        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        colors = sns.color_palette("colorblind")  # , n_colors=3)
        colors = sns.color_palette("deep")  # , n_colors=3)
        # colors = sns.color_palette("pastel")#, n_colors=3)

        # points = [(0, 0), (-0.6, -0.8), (0.75, 1.2), (-0.5, 1.1)]
        # points = [(0, 0)]
        points = [(xs[point_index], ys[point_index])]
        for i, pnt in enumerate(points):
            ax.scatter(pnt[0], pnt[1], color=colors[i], marker='o', s=350)

        periods1 = [0.7, 1.35]
        periods2 = [0.7, 1.35, 1.7]
        periods3 = [0.3, 0.7, 1.35]
        periods4 = [0.3, 0.5, 0.8, 1.1]

        add_1D_hypergrid(ax, (0, 0), 2.3, angle=0, points=points, periods=periods1, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 3.2, angle=45, points=points, periods=periods1, colors=colors, aligned_text=True)
        add_1D_hypergrid(ax, (0, 0), 2.3, angle=90, points=points, periods=periods2, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 3.2, angle=135, points=points, periods=periods2, colors=colors, aligned_text=True)
        add_1D_hypergrid(ax, (0, 0), 2.3, angle=180, points=points, periods=periods3, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 3.2, angle=225, points=points, periods=periods3, colors=colors, aligned_text=True)
        add_1D_hypergrid(ax, (0, 0), 2.3, angle=270, points=points, periods=periods4, colors=colors)
        add_1D_hypergrid(ax, (0, 0), 3.2, angle=315, points=points, periods=periods4, colors=colors, aligned_text=True)

        plt.savefig(filename_str % point_index)  # , bbox_inches='tight')
        plt.clf()

    plt.close()

    # Generate a blue-white-red palette:
    color_list = sns.diverging_palette(240, 10, n=9)

    # Generate a brighter green-white-purple palette
    # color_list = sns.diverging_palette(150, 275, s=80, l=55, n=9)

    # Generate a blue-black-red palette
    # color_list = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark")

    # Generate a colormap object
    # cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

    sns.reset_defaults()


# unit period grid per angle
def plot_2D_hypergrid_frames_3(n=100):
    results = glob.glob("out/hypergrid_2D_frames_3_*")
    for filename in results:
        os.remove(filename)
    filename_str = "out/hypergrid_2D_frames_3_%05u.png"

    sns.set()
    # xs, ys = datasets.spiral(100)
    # xs, ys = datasets.circle(n=100, r=1.5)
    xs, ys = datasets.lemiscate(n=n, alpha=1.2)
    ys = 3 * ys

    num_points = len(xs)
    fig = plt.figure(num=1, figsize=(10, 16))  # , constrained_layout=True)

    n_bins = 4
    n_grids = 8
    n_subspace_dims = 1
    n_input_dims = 2
    gridTransform = create_transform_2D_to_1D_hypergrids(n_input_dims=n_input_dims, n_grids=n_grids, n_bins=n_bins,
                                                         n_subspace_dims=n_subspace_dims)

    for point_index in range(num_points):

        # fig, axes = plt.subplots(2, 1, num=1)
        fig, axes = plt.subplots(2, 1, num=1)

        # axes for hypergrid visual
        ax0 = axes[0]

        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.85, hspace=0.4)
        random.seed(6)
        ax0.set_aspect('equal')
        ax0.set_xlim(-2, 2)
        ax0.set_ylim(-2, 2)

        # palettes
        # --------
        # deep, muted, bright, pastel, dark, colorblind

        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        colors = sns.color_palette("colorblind")  # , n_colors=3)
        colors = sns.color_palette("deep")  # , n_colors=3)
        # colors = sns.color_palette("pastel")#, n_colors=3)

        # points = [(0, 0), (-0.6, -0.8), (0.75, 1.2), (-0.5, 1.1)]
        # points = [(0, 0)]
        points = [(xs[point_index], ys[point_index])]
        for i, pnt in enumerate(points):
            ax0.scatter(pnt[0], pnt[1], color=colors[i], marker='o', s=350)

        unit_periods = [1.0]
        periods1 = [0.7, 1.35]
        periods1 = [0.7, 1.35]
        periods2 = [0.7, 1.35, 1.7]
        periods3 = [0.3, 0.7, 1.35]
        periods4 = [0.3, 0.5, 0.8, 1.1]
        box_height = 0.2

        angles = [0, 45, 90, 135, 180, 225, 270, 315]

        add_1D_hypergrid(ax0, (0, 0), 2.3, angle=angles[0], points=points, periods=unit_periods, colors=colors,
                         box_height=box_height)
        add_1D_hypergrid(ax0, (0, 0), 3.2, angle=angles[1], points=points, periods=unit_periods, colors=colors,
                         box_height=box_height, aligned_text=True)
        add_1D_hypergrid(ax0, (0, 0), 2.3, angle=angles[2], points=points, periods=unit_periods, colors=colors,
                         box_height=box_height)
        add_1D_hypergrid(ax0, (0, 0), 3.2, angle=angles[3], points=points, periods=unit_periods, colors=colors,
                         box_height=box_height, aligned_text=True)
        add_1D_hypergrid(ax0, (0, 0), 2.3, angle=angles[4], points=points, periods=unit_periods, colors=colors,
                         box_height=box_height)
        add_1D_hypergrid(ax0, (0, 0), 3.2, angle=angles[5], points=points, periods=unit_periods, colors=colors,
                         box_height=box_height, aligned_text=True)
        add_1D_hypergrid(ax0, (0, 0), 2.3, angle=angles[6], points=points, periods=unit_periods, colors=colors,
                         box_height=box_height)
        add_1D_hypergrid(ax0, (0, 0), 3.2, angle=angles[7], points=points, periods=unit_periods, colors=colors,
                         box_height=box_height, aligned_text=True)

        # axes for gnome visual
        ax1 = axes[1]
        X_gnomes = gridTransform.transform(points).reshape(n_grids, n_bins)
        subspace_periods = gridTransform.subspace_periods

        # for k in range(len(y)):
        #    if y[k] != 0:
        #        ax1.axvline(x=float(float(k) + 0.5), linewidth=1.5, color=colors[y[k] - 1],
        #                   label=float(ref_points[y[k] - 1]), zorder=6)

        cmap = ListedColormap(['white', 'gray'])
        state_data = np.rot90(X_gnomes, k=-1, axes=(1, 0))
        barprops = dict(cmap=cmap, interpolation='nearest', aspect='auto',
                        extent=[0, state_data.shape[1], 0, state_data.shape[0]])
        # img = ax1.imshow(state_data, **barprops)

        # period labels
        labelsizes = [12]
        ax1.tick_params(axis='y', which='minor', labelsize=labelsizes[0])
        ax1.tick_params(axis='y', which='major', length=25, direction='out')
        ax1.tick_params(axis='y', which='minor', length=0)
        mags = list(subspace_periods.reshape(-1))
        mags = ["%.2f" % mag for mag in mags]

        # sns.heatmap(state_data, square=True, linewidths=.5, linecolor='k', cbar=False, cmap=cmap)
        gnome_plot = sns.heatmap(X_gnomes, ax=ax1, square=True, linewidths=.5, linecolor='k', cbar=False, cmap=cmap)  #
        # g = sns.heatmap(X_gnomes, ax=ax1, square=False, linewidths=.5, linecolor='k', cbar=False, cmap=cmap)#

        # hypergrid labels
        text_angles = []
        for angle in angles:
            text_angles.append("%.0f$^\circ$" % angle)
        gnome_plot.set_yticklabels(text_angles, rotation=0, fontdict={'fontsize': 18})

        # bin labels
        gnome_plot.set_xticklabels([k for k in range(n_bins)], fontdict={'fontsize': 18})

        # barprops = dict(cmap='binary', interpolation='nearest', aspect='auto')
        # state_data = np.rot90(X_gnomes, k=-1, axes=(1, 0))
        # img = ax1.imshow(state_data, **barprops)
        # img = ax1.matshow(state_data, **barprops)
        # img = ax1.matshow(X_gnomes)
        # barprops = dict(cmap=cmap, interpolation='nearest', aspect='auto',
        #                extent=[0, X_gnomes.shape[1], 0, X_gnomes.shape[0]])
        # ax1.xaxis.set_major_locator(ticker.NullLocator())
        # ax1.yaxis.set_major_locator(ticker.NullLocator())

        """
        for y in range(state_data.shape[0] + 1):
            print("horizontal:", y)
            ax1.axhline(y, lw=0.5, color='k', zorder=5)
        for x in range(state_data.shape[1] + 1):
            ax1.axvline(x, lw=0.5, color='k', zorder=5)
        """

        # disable x ticks
        # ax1.xaxis.set_major_locator(ticker.NullLocator())
        # ax1.set_xlim((0.5, state_data.shape[1] - 0.5))
        # ax1.set_xlim((-0.5, state_data.shape[1] + 0.5))

        # rest of ticks for y axis
        # ax1.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
        # ax1.yaxis.set_major_formatter(ticker.NullFormatter())

        # mags.reverse()
        # ax1.yaxis.set_minor_locator(ticker.IndexLocator(1, n_bins // 2))
        # ax1.yaxis.set_minor_locator(ticker.IndexLocator(1, 1))
        # ax1.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))
        ax1.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))
        """

        print(state_data.shape)

        ax1.set_xlim((0.5, state_data.shape[1] - 0.5))
        """

        plt.savefig(filename_str % point_index)  # , bbox_inches='tight')
        plt.clf()

    plt.close()

    # Generate a blue-white-red palette:
    color_list = sns.diverging_palette(240, 10, n=9)

    # Generate a brighter green-white-purple palette
    # color_list = sns.diverging_palette(150, 275, s=80, l=55, n=9)

    # Generate a blue-black-red palette
    # color_list = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark")

    # Generate a colormap object
    # cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

    sns.reset_defaults()


# unit period grid per angle
def plot_2D_hypergrid_frames_4(n=100):
    results = glob.glob("out/hypergrid_2D_frames_4_*")
    for filename in results:
        os.remove(filename)
    filename_str = "out/hypergrid_2D_frames_4_%05u.png"

    sns.set()

    # trajectory for moving point
    xs, ys = datasets.lemiscate(n=n, alpha=1.2)
    ys = 3 * ys

    num_points = len(xs)
    fig = plt.figure(num=1, figsize=(12, 8))  # , constrained_layout=True)

    n_bins = 4
    n_grids = 8
    n_subspace_dims = 1
    n_input_dims = 2
    gridTransform = create_transform_2D_to_1D_hypergrids(n_input_dims=n_input_dims, n_grids=n_grids, n_bins=n_bins,
                                                         n_subspace_dims=n_subspace_dims)

    # deep_cmap = ListedColormap(sns.color_palette("deep"))
    # colors = sns.husl_palette(8, s=.45)
    colors = sns.color_palette("deep", n_colors=n_grids)

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

        points = [(xs[point_index], ys[point_index])]
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
            # grid_colors = [colors[k]]
            grid_colors = [colors[k], colors[k], colors[k]]

            if k % 2 == 1:
                aligned_text = True
            else:
                aligned_text = False

            add_1D_hypergrid(ax0, (0, 0), plot_dist, angle=angles[k], points=points, periods=unit_periods,
                             colors=grid_colors,
                             box_height=box_height, aligned_text=aligned_text)

        # axes for gnome visual
        ax1 = axes[1]
        X_gnomes = gridTransform.transform(points)

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
            curr_color = np.array(colorConverter.to_rgba(colors[k]))
            colors2.append(curr_color)

        cmap = ListedColormap(colors2)
        annot = np.zeros(shape=X_gnomes.shape)
        for k in range(n_bins):
            annot[:, k] = k

        gnome_plot = sns.heatmap(X_gnomes, ax=ax1, square=True, linewidths=0.05, linecolor='k', cbar=False,
                                 cmap=colors2, annot=annot, annot_kws={'color': 'k'}, clip_on=False)

        # X-Axis bin labels
        gnome_plot.set_xticklabels([])
        # gnome_plot.set_xticklabels([k for k in range(n_bins)], fontdict={'fontsize': 12})

        # Y-Axis hypergrid angle labels
        text_angles = []
        for angle in angles:
            text_angles.append("%.0f$^\circ$" % angle)
        gnome_plot.set_yticklabels(text_angles, rotation=0, fontdict={'fontsize': 12})
        gnome_plot.set_aspect('equal')

        plt.savefig(filename_str % point_index)  # , bbox_inches='tight')
        plt.clf()

    plt.close()

    sns.reset_defaults()


# multiple period grids per angle
def plot_2D_hypergrid_frames_5(n=100):
    results = glob.glob("out/hypergrid_2D_frames_5_*")
    for filename in results:
        os.remove(filename)
    filename_str = "out/hypergrid_2D_frames_5_%05u.png"

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
    #print(uniform_periods)

    for k in range(n_angles):
        all_periods += uniform_periods
    #print(all_periods)
    # dist_periods = [0.5 for k in range(n_grids)]

    gridTransform = create_transform_2D_to_1D_hypergrids(n_input_dims=n_input_dims, n_grids=n_grids, n_bins=n_bins,
                                                         n_subspace_dims=n_subspace_dims, periods=all_periods)

    # deep_cmap = ListedColormap(sns.color_palette("deep"))
    # colors = sns.husl_palette(8, s=.45)
    colors = sns.color_palette("deep", n_colors=n_grids)

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

        points = [(xs[point_index], ys[point_index])]
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

        # axes for gnome visual
        ax1 = axes[1]
        X_gnomes = gridTransform.transform(points)

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

        gnome_plot.yaxis.set_major_locator(ticker.IndexLocator(n_grids_per_angle,n_grids_per_angle/2))
        gnome_plot.set_yticklabels(text_angles, rotation=0, fontdict={'fontsize': 12})
        gnome_plot.set_aspect('equal')

        plt.savefig(filename_str % point_index)  # , bbox_inches='tight')
        plt.clf()

    plt.close()

    sns.reset_defaults()


def colorline(
        x, y, z=None, ax=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    # Create an alpha channel of linearly increasing values moving to the right.
    alphas = np.ones(len(x))
    alphas[:, 30:] = np.linspace(1, 0, 70)
    #colors[..., -1] = alphas

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    if not ax is None:
        ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


# multiple period grids per angle
def plot_2D_hypergrid_frames_6(n=100):


    results = glob.glob("out/hypergrid_2D_frames_6_*")
    for filename in results:
        os.remove(filename)
    filename_str = "out/hypergrid_2D_frames_6_%05u.png"

    sns.set()

    # trajectory for moving point
    xs, ys = datasets.lemiscate(n=n, alpha=1.2)
    ys = 3 * ys


    num_points = len(xs)
    fig = plt.figure(num=1, figsize=(12, 8))  # , constrained_layout=True)

    n_bins = 4
    n_subspace_dims = 1
    n_input_dims = 2
    n_angles = 1
    n_grids_per_angle = 1
    n_grids = n_angles * n_grids_per_angle

    all_periods = []
    # dist_periods = [0.5, 1.0, 1.5]
    uniform_periods = list(np.linspace(0, 2, endpoint=False, num=(n_grids_per_angle + 1))[1:])
    #print(uniform_periods)

    for k in range(n_angles):
        all_periods += uniform_periods
    #print(all_periods)
    # dist_periods = [0.5 for k in range(n_grids)]

    gridTransform = create_transform_2D_to_1D_hypergrids(n_input_dims=n_input_dims, n_grids=n_grids, n_bins=n_bins,
                                                         n_subspace_dims=n_subspace_dims, periods=all_periods)

    # deep_cmap = ListedColormap(sns.color_palette("deep"))
    # colors = sns.husl_palette(8, s=.45)
    colors = sns.color_palette("deep", n_colors=n_grids)

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

        points = [(xs[point_index], ys[point_index])]
        for i, pnt in enumerate(points):
            ax0.scatter(pnt[0], pnt[1], color='k', marker='o', s=30)

        unit_periods = [1.0]
        box_height = 0.2
        #dist_periods = [0.5, 1.0, 1.5]
        #angles = [0, 45, 90, 135, 180, 225, 270, 315]
        dist_periods = [1]
        angles = [0]

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
        #path = mpath.Path(np.column_stack([xs, ys]))
        #verts = path.interpolated(steps=3).vertices
        #x, y = verts[:, 0], verts[:, 1]
        #z = np.linspace(0, 1, len(x))
        #lc = colorline(x, y, z, ax=ax0, cmap=plt.get_cmap('jet'), linewidth=2)

        # axes for gnome visual
        ax1 = axes[1]
        X_gnomes = gridTransform.transform(points)

        ref_points =[(0,0), (1,1)]
        ref_gnomes = gridTransform.transform(ref_points)
        ref_gnomes[:,0:3] = True


        otsuka_result = otsuka_similarity(X_gnomes.reshape(len(points), -1), ref_gnomes.reshape(len(ref_points), -1))
        tanimoto_result = tanimoto_similarity(X_gnomes.reshape(len(points), -1), ref_gnomes.reshape(len(ref_points), -1))
        gnome_result = gnome_similarity(X_gnomes.reshape(len(points), -1), ref_gnomes.reshape(len(ref_points), -1))
        weighted_result = weighted_similarity(X_gnomes.reshape(len(points), -1), ref_gnomes.reshape(len(ref_points), -1))

        print("points:", points)
        print("Similarities")
        print("gnome, tanimoto, otsuka, weighted")
        print(gnome_result, tanimoto_result, otsuka_result, weighted_result)
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

        gnome_plot.yaxis.set_major_locator(ticker.IndexLocator(n_grids_per_angle,n_grids_per_angle/2))
        gnome_plot.set_yticklabels(text_angles, rotation=0, fontdict={'fontsize': 12})
        gnome_plot.set_aspect('equal')

        plt.savefig(filename_str % point_index)  # , bbox_inches='tight')
        plt.clf()

        #break

    plt.close()

    sns.reset_defaults()

if __name__ == "__main__":
    # plot_2D_hypergrid_frames_1(n=1)
    # plot_2D_hypergrid_frames_2(n=200)
    # plot_2D_hypergrid_frames_2(n=40)
    # plot_2D_hypergrid_frames_3(n=200)
    # plot_2D_hypergrid_frames_4(n=200)
    #plot_2D_hypergrid_frames_5(n=200)
    #plot_2D_hypergrid_frames_6(n=200)
    plot_2D_hypergrid_frames_6(n=1)

