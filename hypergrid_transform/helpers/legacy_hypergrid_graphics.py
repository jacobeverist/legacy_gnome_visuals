import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import matplotlib.patches as patches

from matplotlib.transforms import Affine2D


def add_1D_hypergrid_artist(ax, origin, displacement_vector, n_bins=4, bin_period=0.7, grid_period=3.0,
                            aligned_text=False, colors=None):
    rotation_point = origin
    origin_x, origin_y = Affine2D().transform(
        [rotation_point[0] + displacement_vector[0], rotation_point[1] + displacement_vector[1]])

    print(origin_x, origin_y)


def add_1D_hypergrid(ax, origin, projection_length, points=(), n_bins=4, periods=(0.7, 1.35), angle=0, grid_range=3.0,
                     aligned_text=False, colors=None):
    rotation_point = origin

    projection_angle = angle + 90

    # the projected starting point for grid rectangles
    projection_point_rotation = Affine2D().rotate_deg_around(rotation_point[0], rotation_point[1], projection_angle)
    origin_x, origin_y = projection_point_rotation.transform([rotation_point[0] + projection_length, rotation_point[1]])

    n_grids = len(periods)

    x_high = grid_range / 2
    x_low = -grid_range / 2

    box_height = 0.1

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

        ax.plot([pnt[0], line_x], [pnt[1], line_y], clip_on=False, color=colors[iter_index])  # , zorder=6)

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
        # interval_count = int((x_low - bin_interval_x) // bin_interval_x)+1
        # while interval_count * bin_interval_x <= x_high + bin_interval_x:
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

    # regularize angle between +/- 180
    text_angle = angle
    if angle > 0:
        while text_angle > 180:
            text_angle -= 360
    elif angle < 0:
        while text_angle < -180:
            text_angle += 360

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

    ax.text(label_x, label_y, "%.1f$^\circ$" % text_angle, rotation=transform_angle, rotation_mode='anchor',
            fontsize=16, va='center', ha='center', clip_on=False)


def add_text_rect(ax, box_x, box_y, box_width, box_height, angle=0, linewidth=1.5, edgecolor='k', fontsize=8,
                  facecolor='none', text_str=None, aligned_text=False):
    # text_offset = -0.01
    text_offset = 0.0

    # data space coordinates to find new point after rotated
    fixed_point_rotation = Affine2D().rotate_deg_around(box_x, box_y, angle)

    # add rectangle at corner position and rotate by angle
    rect = patches.Rectangle((box_x, box_y), box_width, box_height, angle=angle, linewidth=linewidth,
                             edgecolor=edgecolor,
                             facecolor=facecolor, clip_on=False)
    ax.add_patch(rect)

    # center of rectangle for text box
    rect_center_pos = [box_x + box_width / 2, box_y + box_height * 0.5 + text_offset]

    # rotate around rectangle corner
    text_pos = fixed_point_rotation.transform(rect_center_pos)

    if aligned_text:
        text_angle = angle

        if angle > 0:
            while text_angle > 180:
                text_angle -= 360
        elif angle < 0:
            while text_angle < -180:
                text_angle += 360

        if abs(text_angle) > 90:
            text_angle += 180
    else:
        text_angle = 0

    # add text box to center of rectangle
    ax.text(text_pos[0], text_pos[1], text_str, rotation=text_angle, rotation_mode='anchor',
            fontsize=fontsize, va='center', ha='center', clip_on=False)

if __name__ == "__main__":
    from matplotlib import ticker
    import random

    fig = plt.figure(num=1, figsize=(12, 8))  # , constrained_layout=True)
    fig, axes = plt.subplots(1, 2, num=1, gridspec_kw={"width_ratios": [1, 0.4]})

    # axes for hypergrid visual
    ax0 = axes[0]

    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=1)
    # fig.subplots_adjust(wspace=0.8)

    ax0.set_aspect('equal')
    ax0.set_xlim(-2, 2)
    ax0.set_ylim(-2, 2)
    ax0.tick_params(labelsize=8)
    tick_locs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    ax0.xaxis.set_major_locator(ticker.FixedLocator(tick_locs))
    ax0.yaxis.set_major_locator(ticker.FixedLocator(tick_locs))

    add_1D_hypergrid(ax0, (0, 0), 2.0, angle=0.0, points=[0.0,], periods=[1.0],
                     colors=grid_colors,
                     box_height=box_height, aligned_text=aligned_text)

    plt.savefig("visual_test01.png")
    plt.clf()

# plot fading line