import networkx as nx
from intervals import FloatInterval as I
from intervals.exc import *
import numpy as np
import pprint
import itertools as it
import string

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.transforms import Affine2D
from matplotlib.colors import ListedColormap, colorConverter
from matplotlib.lines import Line2D
import matplotlib.patches as patches


# do analysis of hypergrids by considering them as sets of intervals
# perform intersections of intervals and produce the interval graph
# all k-cliques of interval nodes, where k = number of grids, are the list of possible states of interval occupancy
# assuming only over the unit interval and all grid periods = 1

# Possible starting intervals for intersection or union
# I.open(-I.inf, I.inf)
# unit_interval
# I.closed(0,0)
# I.empty()
# None


def add_1D_hypergrid(ax, projection_length=0, origin=(0, 0), points=(), bins=(4, 4), periods=(0.7, 1.35), angle=0,
                     grid_range=3.0,
                     box_height=0.1, minimal_range=False, boundary_on_origin=False,
                     aligned_text=False, colors=None, grid_label_templates=None):
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

        n_bins = bins[grid_i]

        # grid magnitude
        period = periods[grid_i]

        # interval length of bins
        bin_interval_x = period / n_bins

        # position of bottom left corner of the first bin rectangle of this grid
        if boundary_on_origin:
            grid_x = origin_x
        else:
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

        # draw the bins only once at the origin
        if minimal_range:
            x_low = 0
            x_high = period

        # index of first bin within specified range
        interval_count = int((x_low - bin_interval_x) // bin_interval_x) + 1

        # draw bins until reach end of range
        while interval_count * bin_interval_x < x_high:

            # bottom left corner of current bin rectangle
            bin_boundary_x = grid_x + bin_interval_x * interval_count
            bin_boundary_y = grid_y

            # bottom left corner after its been rotated
            corner_pos = fixed_point_rotation.transform([bin_boundary_x, bin_boundary_y])

            num_occs = 0
            # total_color = 'none'
            total_color = np.array(colorConverter.to_rgba(colors[grid_i]))

            """
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
            """
            alpha = 1.0
            linewidth = 1.5
            if interval_count < 0:
                m = abs(bin_interval_x * interval_count)
                alpha = 0.3 - m * 0.15
                linewidth = 0.5
            elif interval_count >= n_bins:
                m = abs(bin_interval_x * interval_count)
                alpha = 0.3 - m * 0.15
                linewidth = 0.5
            if alpha < 0:
                alpha = 0

            if num_occs > 0:
                total_color = total_color / num_occs

            # create rectangle with text inside
            if grid_label_templates is None:
                add_text_rect(ax, corner_pos[0], corner_pos[1], bin_interval_x, box_height, angle=angle,
                              text_str=str(interval_count % n_bins), aligned_text=aligned_text, facecolor=total_color,
                              alpha=alpha, linewidth=linewidth)
            else:
                add_text_rect(ax, corner_pos[0], corner_pos[1], bin_interval_x, box_height, angle=angle,
                              text_str=grid_label_templates[grid_i] % (interval_count % n_bins,),
                              aligned_text=aligned_text, facecolor=total_color,
                              alpha=alpha, linewidth=linewidth)

            # increment counter
            interval_count += 1


def add_1D_scalar_encoder(ax, projection_length=0, origin=(0, 0), points=(), bins=(4, 4), periods=(0.7, 1.35), angle=0,
                          grid_range=3.0,
                          box_height=0.1, minimal_range=False, boundary_on_origin=False,
                          aligned_text=False, colors=None, grid_label_templates=None):
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

        n_bins = bins[grid_i]

        # grid magnitude
        period = periods[grid_i]

        # interval length of bins
        bin_interval_x = period / n_bins

        # position of bottom left corner of the first bin rectangle of this grid
        if boundary_on_origin:
            grid_x = origin_x
        else:
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

        # draw the bins only once at the origin
        if minimal_range:
            x_low = 0
            x_high = period

        # index of first bin within specified range
        interval_count = int((x_low - bin_interval_x) // bin_interval_x) + 1

        # draw bins until reach end of range
        while interval_count * bin_interval_x < x_high:

            # bottom left corner of current bin rectangle
            bin_boundary_x = grid_x + bin_interval_x * interval_count
            bin_boundary_y = grid_y

            # bottom left corner after its been rotated
            corner_pos = fixed_point_rotation.transform([bin_boundary_x, bin_boundary_y])

            num_occs = 0
            # total_color = 'none'
            total_color = np.array(colorConverter.to_rgba(colors[grid_i]))

            """
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
            """
            alpha = 1.0
            linewidth = 1.5
            if interval_count < 0:
                m = abs(bin_interval_x * interval_count)
                alpha = 0.3 - m * 0.15
                linewidth = 0.5
            elif interval_count >= n_bins:
                m = abs(bin_interval_x * interval_count)
                alpha = 0.3 - m * 0.15
                linewidth = 0.5
            if alpha < 0:
                alpha = 0

            if num_occs > 0:
                total_color = total_color / num_occs

            # create rectangle with text inside
            if grid_label_templates is None:
                add_text_rect(ax, corner_pos[0], corner_pos[1], bin_interval_x, box_height, angle=angle,
                              text_str=str(interval_count % n_bins), aligned_text=aligned_text, facecolor=total_color,
                              alpha=alpha, linewidth=linewidth)
            else:
                add_text_rect(ax, corner_pos[0], corner_pos[1], bin_interval_x, box_height, angle=angle,
                              text_str=grid_label_templates[grid_i] % (interval_count % n_bins,),
                              aligned_text=aligned_text, facecolor=total_color,
                              alpha=alpha, linewidth=linewidth)

            # increment counter
            interval_count += 1


def add_text_rect(ax, box_x, box_y, box_width, box_height, angle=0, linewidth=1.5, edgecolor='k', fontsize=8,
                  facecolor='none', text_str=None, aligned_text=False, alpha=1.0, text_v_offset=-0.01):
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


def create_k_overlaps_of_w(k, w, interval_to_partition, is_debug=False):
    if is_debug:
        print("%d bins, w=%d" % (k, w))
        print("------------------")

    upper_bound = interval_to_partition.upper
    lower_bound = interval_to_partition.lower

    interval_size = upper_bound - lower_bound
    num_partitions = k - (w - 1)
    step_size = 1 / num_partitions * interval_size
    bin_size = step_size * w
    if is_debug:
        print("%.2f interval size" % interval_size)
        print("%d interval partitions" % num_partitions)
        print("%.2f step_size" % step_size)
        print("%.2f bin_size" % bin_size)
        print("------------------")

    equidist_points = np.linspace(lower_bound, upper_bound, endpoint=True, num=num_partitions + 1)

    val_list = list(equidist_points)

    # prepend interval boundaries out-of-bounds
    for i in range(w - 1):
        new_boundary = val_list[0] - step_size
        val_list = [new_boundary, ] + val_list

    # append interval boundaries out-of-bounds
    for i in range(w - 1):
        new_boundary = val_list[-1] + step_size
        val_list = val_list + [new_boundary, ]

    if is_debug:
        print("boundary points:", val_list)
        # print(type(np_points), np_points)

    intervals = [I.closed_open(val_list[c], val_list[c + w]) for c in range(0, len(val_list) - w)]

    # intervals = [I.closed_open(val_list[c - 1], val_list[c]) for c in range(1, len(val_list))]
    if is_debug:
        print("partition:", intervals)

    return intervals


def create_k_partition(k, interval_to_partition, is_debug=False):
    if is_debug:
        print("%d equal size components" % k)
        print("------------------")

    upper_bound = interval_to_partition.upper
    lower_bound = interval_to_partition.lower
    # equidist_points = I.iterate(interval_to_partition, incr=((upper_bound - lower_bound) / k))
    equidist_points = np.linspace(lower_bound, upper_bound, endpoint=True, num=k + 1)

    val_list = list(equidist_points)
    if is_debug:
        print("boundary points:", val_list)
        # print(type(np_points), np_points)

    intervals = [I.closed_open(val_list[c - 1], val_list[c]) for c in range(1, len(val_list))]
    if is_debug:
        print("partition:", intervals)

    return intervals


def iter_all_combinations(partition_dict):
    partition_keys = list(partition_dict.keys())
    partition_keys.sort()

    partition_intervals = []
    for k in partition_keys:
        bin_intervals = []

        for bin, interval in enumerate(partition_dict[k]):
            bin_intervals.append((bin, interval))

        partition_intervals.append(bin_intervals)

    return it.product(*partition_intervals)


def iter_all_w_combinations(partition_dict, w=1):
    partition_keys = list(partition_dict.keys())
    partition_keys.sort()

    combo_intervals = []
    for k in partition_keys:
        bin_intervals = []
        for bin, interval in enumerate(partition_dict[k]):
            bin_intervals.append((bin, interval))

        w_overlaps = []
        for overlap_combo in range(len(bin_intervals) - w):
            w_overlaps.append((overlap_combo, bin_intervals[overlap_combo:overlap_combo + w]))

        # partition_intervals.append(partition_dict[k])
        combo_intervals.append(w_overlaps)

    return it.product(*combo_intervals)


def union_intervals(interval_list):
    # start empty interval
    total_interval = interval_list[0]
    for c in range(1, len(interval_list)):
        total_interval = total_interval | interval_list[c]

    return total_interval



def intersect_intervals(interval_list):
    if not len(interval_list) > 0:
        return I.from_string('(0, 0]')
        #return I.empty()

    overlapped_interval = interval_list[0]
    for interval in interval_list[1:]:
        try:
            overlapped_interval = overlapped_interval & interval
        except IllegalArgument:
            # return quickly if its already empty
            return I.from_string('(0, 0]')
            #return I.empty()

        #print("overlapped:", overlapped_interval)

        # return quickly if its already empty
        if overlapped_interval.empty:
            return I.from_string('(0, 0]')
            #return I.empty()

    return overlapped_interval


def find_self_intersected_combinations(k_partitions, w=1):
    interval_combinations = []

    print("combinations")
    for comb in iter_all_w_combinations(k_partitions, w=w):
        print(comb)

    for comb in iter_all_w_combinations(k_partitions, w=w):
        print(comb)

        intervals = []
        bins = []
        for bin, overlap_intervals in comb:

            for interval in overlap_intervals:
                intervals.append(interval)
            bins.append(bin)

        intersected_interval = intersect_intervals(intervals)

        if not intersected_interval.empty:
            interval_size = intersected_interval.upper - intersected_interval.lower

            interval_combinations.append((interval_size, intersected_interval, intervals, tuple(bins)))

    return interval_combinations


def find_intersected_combinations(k_partitions):
    interval_combinations = []
    for comb in iter_all_combinations(k_partitions):

        intervals = []
        bins = []
        for bin, interval in comb:
            intervals.append(interval)
            bins.append(bin)

        intersected_interval = intersect_intervals(intervals)

        if not intersected_interval.empty:
            interval_size = intersected_interval.upper - intersected_interval.lower

            interval_combinations.append((interval_size, intersected_interval, intervals, tuple(bins)))

    return interval_combinations


def find_contained_interval_combination(x_val, partition_dict):
    partition_keys = list(partition_dict.keys())
    partition_keys.sort()

    contained_intervals = []

    for k in partition_keys:

        partition_list = partition_dict[k]
        for interval_index, interval in enumerate(partition_list):
            if x_val in interval:
                contained_intervals.append((k, interval_index, interval))
                break

    return contained_intervals


def interval_to_string(interval):
    return str(interval) #, conv=lambda v: "%.2f" % v)
    #return I.to_string(interval, conv=lambda v: "%.2f" % v)


def create_interval_graph(interval_combinations):
    if len(interval_combinations) == 0:
        return None

    interval_size, intersected_interval, comb, bins = interval_combinations[0]

    # number of grids
    n_grids = len(bins)
    grid_names = string.ascii_uppercase[:n_grids]

    node_dict = {}
    edge_list = []
    for interval_size, intersected_interval, comb, bins in interval_combinations:

        labeled_bins = [grid_names[grid_i] + str(bins[grid_i]) for grid_i in range(len(bins))]
        # print(labeled_bins)
        # print("combination:", list(it.combinations(labeled_bins, 2)))
        edges = list(it.combinations(labeled_bins, 2))
        edge_list += edges
        for grid_i in range(n_grids):
            bin_i = bins[grid_i]
            node_name = grid_names[grid_i] + str(bin_i)
            node_dict[node_name] = 1

    node_list = list(node_dict.keys())
    node_list.sort()
    # print(node_list)

    G = nx.Graph()
    G.add_nodes_from(node_list)

    G.add_edges_from(edge_list)

    # G.add_edge(node_list[0], node_list[1])

    return G


def plot_interval_graph_1(partition_sizes=(2, 3), lower_bound=0, upper_bound=1, file_name=None):
    unit_interval = I.closed(lower_bound, upper_bound)

    k_partitions = {}

    # create identically sized partition of the unit interval into k components
    for k in partition_sizes:
        k_partitions[k] = create_k_partition(k, unit_interval, is_debug=False)

    # print("list of partitions by k connected components:")
    # pp = pprint.PrettyPrinter()
    # pp.pprint(k_partitions)
    # print()

    # print("combinatoric search of possible intersections")
    interval_combinations = find_intersected_combinations(k_partitions)
    # print("%d total combinations" % len(interval_combinations))

    n_grids = len(interval_combinations[0][3])
    grid_names = string.ascii_uppercase[:n_grids]

    # grid_deltas = [0]
    # prev_bins = list([0 for i in range(n_grids)])

    boundaries = [0]
    # grid_deltas = [-1]
    grid_deltas = []
    grid_delta_counts = []
    prev_bins = None

    for interval_size, intersected_interval, comb, bins in interval_combinations:
        # print("%.3f" % interval_size, intersected_interval, comb, bins)
        # print(bins, "%.3f" % interval_size, interval_to_string(intersected_interval))
        boundaries.append(intersected_interval.upper)

        if not prev_bins is None:
            grid_delta_index = None
            grid_delta_count = 0
            grids = list(range(n_grids))
            grids.reverse()
            for i in grids:
                if bins[i] != prev_bins[i]:
                    grid_delta_index = i
                    grid_delta_count += 1

            grid_deltas.append(grid_delta_index)
            grid_delta_counts.append(grid_delta_count)

        prev_bins = bins

    # grid_deltas.append(-1)

    # print("deltas:", grid_deltas)
    # print("counts:", grid_delta_counts)

    # print(boundaries[:-1])
    # print(boundaries)

    x_vals = [boundaries[0]]
    y_vals = [boundaries[0]]
    for j in range(1, len(boundaries)):
        x_vals.append(boundaries[j - 1])
        y_vals.append(boundaries[j])

        x_vals.append(boundaries[j])
        y_vals.append(boundaries[j])

    x_point_vals = []
    y_point_vals = []
    for j in range(0, len(boundaries)):
        x_point_vals.append(boundaries[j])
        y_point_vals.append(boundaries[j])

    # print()
    # print("boundaries:", boundaries)
    # print()

    G = create_interval_graph(interval_combinations)
    # print(G)

    # pos = nx.graphviz_layout(G)
    # print("nodes:", G.nodes)

    partition_keys = list(k_partitions.keys())
    partition_keys.sort()

    partition_intervals = []
    for i in range(len(partition_keys)):
        bin_intervals = []
        key = partition_keys[i]
        for bin, interval in enumerate(k_partitions[key]):
            bin_intervals.append((bin, interval))
        partition_intervals.append(bin_intervals)
    # print(partition_intervals[0][0][1])

    pos = nx.kamada_kawai_layout(G)

    # separation of nodes into groups
    shells = [[] for i in range(n_grids)]
    # colors = sns.color_palette("deep", n_colors=n_grids)
    colors = sns.color_palette("muted", n_colors=n_grids)
    node_colors = []  # colors[k] for node in G.nodes]
    for node in G.nodes:
        shell_i = grid_names.find(node[0])
        interval_i = int(node[1:])
        shells[shell_i].append(node)
        node_colors.append(colors[shell_i])
        pos[node][1] = shell_i

        # print("interval_i =", interval_i)

        intvl = partition_intervals[shell_i][interval_i][1]
        mid_x = intvl.upper - (intvl.upper - intvl.lower) / 2.0
        # print("mid:", mid_x)
        pos[node][0] = mid_x

    # pos = nx.shell_layout(G, shells)
    # pos = nx.bipartite_layout(G, shells[0])
    # pos = nx.fruchterman_reingold_layout(G)
    # pos = nx.spectral_layout(G)
    # pos = nx.planar_layout(G)
    # print("case1")
    # pos = nx.kamada_kawai_layout(G)
    # print("case2")
    # print("layout:", type(pos), pos)

    # for node in G.nodes:
    #    print(node)

    fig = plt.figure(num=1, figsize=(10, 8), constrained_layout=True)
    fig, axes = plt.subplots(2, 1, num=1)
    ax0 = axes[0]
    ax1 = axes[1]

    # H = nx.path_graph(4)
    # shells = [[0], [1, 2, 3]]
    # pos = nx.shell_layout(H, shells)
    # node_color = # : color string, or array of floats, (default='#1f78b4')
    # nx.draw(G, pos, with_labels=True, ax=ax0, node_color=node_colors)
    # nx.draw_networkx(G, pos, with_labels=True, ax=ax0, alpha=0.5, font_size=8, node_color=node_colors)
    # nx.draw_networkx(G, pos, with_labels=True, ax=ax0, font_weight='bold', font_size=8, node_color=node_colors)
    nx.draw_networkx(G, pos, with_labels=True, ax=ax0, font_size=8, node_color=node_colors)
    # ax0.set_axis_off()
    ax0.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True)
    ax0.set_title("Interval Graph")
    ax0.set_xlim(-0.1, 1.1)
    # ax0.xaxis.set_major_locator(ticker.NullLocator())
    # ax0.yaxis.set_major_locator(ticker.NullLocator())

    # doesn't work!
    tick_locs = [0, 0.5, 1]
    ax1.set_xlim(-0.1, 1.1)
    ax1.xaxis.set_major_locator(ticker.FixedLocator(tick_locs))
    ax1.yaxis.set_major_locator(ticker.FixedLocator(tick_locs))
    ax1.set_axis_on()
    ax1.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True)
    # ax1.set_title("Boundary Points\nColor by Transitioning Grid, Size by Number of Simultaneous Grid Transitions")
    ax1.set_title("Boundary Points\nColor / Size = Transitioning Grid / Transition Count")
    ax1.set_xlabel("Actual Value")
    ax1.set_ylabel("Represented Value")

    ax1.plot(x_vals, y_vals, color='k')

    unit_x_limits = [x_point_vals[0], x_point_vals[-1]]
    unit_y_limits = [y_point_vals[0], y_point_vals[-1]]

    ax1.scatter(unit_x_limits, unit_y_limits, c='k', zorder=10)

    boundary_x = x_point_vals[1:-1]
    boundary_y = y_point_vals[1:-1]
    boundary_colors = []
    for k in range(len(grid_deltas)):
        boundary_colors.append(colors[grid_deltas[k]])

    delta_sizes = [grid_delta_counts[i] * 100 for i in range(len(grid_delta_counts))]
    all_labels = [str(partition_sizes[j]) + " intervals" for j in range(len(partition_sizes))]
    delta_labels = [all_labels[grid_deltas[i]] for i in range(len(grid_delta_counts))]
    for grid_i in range(n_grids):
        grid_color = colors[grid_i]
        grid_label = str(partition_sizes[grid_i]) + " intervals"
        grid_sizes = []

        grid_x = []
        grid_y = []
        indices = []
        for j in range(len(grid_deltas)):
            if grid_deltas[j] == grid_i:
                indices.append(j)
                grid_x.append(boundary_x[j])
                grid_y.append(boundary_y[j])
                grid_sizes.append((grid_delta_counts[j] * 10) ** 2)

        ax1.scatter(grid_x, grid_y, color=grid_color, s=grid_sizes, label=grid_label, zorder=10)

    # ax1.scatter(boundary_x, boundary_y, c=boundary_colors, s=delta_sizes, label=delta_labels, zorder=10)
    # for j in range(len(boundary_x)):
    #    ax1.scatter(boundary_x[j], boundary_y[j], color=boundary_colors[j], s=delta_sizes[j], label=delta_labels[j],
    #                zorder=10)

    title_str = "%d Grids, Number of Intervals: " % len(partition_sizes)
    for j in range(len(partition_sizes) - 1):
        title_str += "%d / " % partition_sizes[j]
    title_str += "%d" % partition_sizes[-1]

    fig.suptitle(title_str)
    # fig.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=1)
    # fig.subplots_adjust(hspace=0.5)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title="Partitions", loc='upper right',
               bbox_to_anchor=(1.15, 0.4))  # , fontsize=20)

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.savefig("output.png", bbox_inches='tight')
    plt.clf()


def plot_interval_graph_2(partition_sizes=(2, 3), lower_bound=0, upper_bound=1, file_name=None):
    unit_interval = I.closed(lower_bound, upper_bound)

    k_partitions = {}

    # create identically sized partition of the unit interval into k components
    for k in partition_sizes:
        k_partitions[k] = create_k_partition(k, unit_interval, is_debug=False)

    print("list of partitions by k connected components:")
    pp = pprint.PrettyPrinter()
    pp.pprint(k_partitions)
    print()

    print("combinatoric search of possible intersections")
    interval_combinations = find_intersected_combinations(k_partitions)
    print("%d total combinations" % len(interval_combinations))

    n_grids = len(interval_combinations[0][3])
    grid_names = string.ascii_uppercase[:n_grids]

    boundaries = [0]
    grid_deltas = []
    grid_delta_counts = []
    prev_bins = None

    for interval_size, intersected_interval, comb, bins in interval_combinations:
        # print("%.3f" % interval_size, intersected_interval, comb, bins)
        # print(bins, "%.3f" % interval_size, interval_to_string(intersected_interval))
        boundaries.append(intersected_interval.upper)

        if not prev_bins is None:
            grid_delta_index = None
            grid_delta_count = 0
            grids = list(range(n_grids))
            grids.reverse()
            for i in grids:
                if bins[i] != prev_bins[i]:
                    grid_delta_index = i
                    grid_delta_count += 1

            grid_deltas.append(grid_delta_index)
            grid_delta_counts.append(grid_delta_count)

        prev_bins = bins

    x_vals = [boundaries[0]]
    y_vals = [boundaries[0]]
    for j in range(1, len(boundaries)):
        x_vals.append(boundaries[j - 1])
        y_vals.append(boundaries[j])

        x_vals.append(boundaries[j])
        y_vals.append(boundaries[j])

    x_point_vals = []
    y_point_vals = []
    for j in range(0, len(boundaries)):
        x_point_vals.append(boundaries[j])
        y_point_vals.append(boundaries[j])

    colors = sns.color_palette("muted", n_colors=n_grids)

    fig = plt.figure(num=1, figsize=(10, 8), constrained_layout=True)
    fig, axes = plt.subplots(3, 1, num=1, sharex=True)

    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]

    boundary_x = x_point_vals[1:-1]
    boundary_y = y_point_vals[1:-1]
    boundary_colors = []
    for k in range(len(grid_deltas)):
        boundary_colors.append(colors[grid_deltas[k]])

    all_labels = [str(partition_sizes[j]) + " intervals" for j in range(len(partition_sizes))]
    for grid_i in range(n_grids):
        grid_sizes = []

        grid_x = []
        grid_y = []
        indices = []
        for j in range(len(grid_deltas)):
            if grid_deltas[j] == grid_i:
                indices.append(j)
                grid_x.append(boundary_x[j])
                grid_y.append(boundary_y[j])
                grid_sizes.append((grid_delta_counts[j] * 10) ** 2)

    title_str = "One-Hot Encoders on Interval (0,1)\nBins: "
    for j in range(len(partition_sizes) - 1):
        title_str += "%d / " % partition_sizes[j]
    title_str += "%d" % partition_sizes[-1]

    fig.suptitle(title_str)

    # draw the grids
    ax0.set_xlim(-0.1, 1.1)
    ax0.set_ylim(-1, n_grids + 1)
    ax0.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        labelbottom=False,
        labelleft=False)
    ax0.set_title("Encoder Intervals")

    ax1.set_ylim(-1, 2)
    ax1.set_axis_on()
    ax1.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        labelbottom=False,
        labelleft=False)
    ax1.set_title("Granulation of Space")

    # number of points
    num_points = len(x_point_vals)

    for j in range(1, num_points):
        origin_point = (x_point_vals[j - 1], 0.0)
        width = x_point_vals[j] - x_point_vals[j - 1]
        height = 1

        # create rectangle with text inside
        add_text_rect(ax1, origin_point[0], origin_point[1], width, height, angle=0, linewidth=1.5, edgecolor='k',
                      fontsize=8, facecolor='none', text_str="%.2f" % width, alpha=1.0,
                      text_v_offset=min(-0.1, -height / 1.2))

    grid_label_templates = [prefix + "%d" for prefix in grid_names]

    dist_periods = [1.0 for j in range(n_grids)]
    grid_colors = [colors[j] for j in range(n_grids)]
    add_1D_hypergrid(ax0, bins=partition_sizes, periods=dist_periods, minimal_range=True, boundary_on_origin=True,
                     colors=grid_colors, box_height=1, grid_label_templates=grid_label_templates)

    ax2.vlines(boundary_x, ymin=0, ymax=grid_delta_counts, color='k')
    ax2.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True)
    ax2.yaxis.set_major_locator(ticker.IndexLocator(1, 0))

    ax2.set_ylim(bottom=0)
    ax2.set_title("Change-Point Count")
    ax2.set_xlabel("Unit Interval")

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')


def plot_interval_graph_3(partition_sizes=(2, 3), lower_bound=0, upper_bound=1, file_name=None):
    unit_interval = I.closed(lower_bound, upper_bound)

    k_partitions = {}

    # 2 independent, 1 dependent
    # n = number of bins
    # w = number of adjacent bins that overlap
    # s = step size, k = partitions, k = 1/s

    # specify k and w
    # generate n bins, starting at [(1-w)*s+i*s, (i+1)*s], for i'th bin

    # create identically sized partition of the unit interval into k components
    for k in partition_sizes:
        k_partitions[k] = create_k_overlaps_of_w(k, min(2, k - 1), unit_interval, is_debug=True)

    # print("list of partitions by k connected components:")
    # pp = pprint.PrettyPrinter()
    # pp.pprint(k_partitions)
    # print()

    interval_combinations = find_self_intersected_combinations(k_partitions, w=min(2, k - 1))
    print(interval_combinations)
    print("%d total combinations" % len(interval_combinations))

    n_grids = len(interval_combinations[0][3])
    grid_names = string.ascii_uppercase[:n_grids]

    boundaries = [0]
    grid_deltas = []
    grid_delta_counts = []
    prev_bins = None

    for interval_size, intersected_interval, comb, bins in interval_combinations:
        boundaries.append(intersected_interval.upper)

        if not prev_bins is None:
            grid_delta_index = None
            grid_delta_count = 0
            grids = list(range(n_grids))
            grids.reverse()
            for i in grids:
                if bins[i] != prev_bins[i]:
                    grid_delta_index = i
                    grid_delta_count += 1

            grid_deltas.append(grid_delta_index)
            grid_delta_counts.append(grid_delta_count)

        prev_bins = bins

    x_vals = [boundaries[0]]
    y_vals = [boundaries[0]]
    for j in range(1, len(boundaries)):
        x_vals.append(boundaries[j - 1])
        y_vals.append(boundaries[j])

        x_vals.append(boundaries[j])
        y_vals.append(boundaries[j])

    x_point_vals = []
    y_point_vals = []
    for j in range(0, len(boundaries)):
        x_point_vals.append(boundaries[j])
        y_point_vals.append(boundaries[j])

    colors = sns.color_palette("muted", n_colors=n_grids)

    fig = plt.figure(num=1, figsize=(10, 8), constrained_layout=True)
    fig, axes = plt.subplots(3, 1, num=1, sharex=True)

    ax1 = axes[1]
    ax2 = axes[2]
    ax0 = axes[0]

    boundary_x = x_point_vals[1:-1]
    boundary_y = y_point_vals[1:-1]
    boundary_colors = []
    for k in range(len(grid_deltas)):
        boundary_colors.append(colors[grid_deltas[k]])

    all_labels = [str(partition_sizes[j]) + " intervals" for j in range(len(partition_sizes))]
    for grid_i in range(n_grids):
        grid_sizes = []

        grid_x = []
        grid_y = []
        indices = []
        for j in range(len(grid_deltas)):
            if grid_deltas[j] == grid_i:
                indices.append(j)
                grid_x.append(boundary_x[j])
                grid_y.append(boundary_y[j])
                grid_sizes.append((grid_delta_counts[j] * 10) ** 2)

    title_str = "One-Hot Encoders on Interval (0,1)\nBins: "
    for j in range(len(partition_sizes) - 1):
        title_str += "%d / " % partition_sizes[j]
    title_str += "%d" % partition_sizes[-1]

    fig.suptitle(title_str)

    # draw the grids
    ax0.set_xlim(-0.1, 1.1)
    ax0.set_ylim(-1, n_grids + 1)
    ax0.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        labelbottom=False,
        labelleft=False)
    ax0.set_title("Encoder Intervals")

    grid_label_templates = [prefix + "%d" for prefix in grid_names]

    dist_periods = [1.0 for j in range(n_grids)]
    grid_colors = [colors[j] for j in range(n_grids)]
    add_1D_hypergrid(ax0, bins=partition_sizes, periods=dist_periods, minimal_range=True, boundary_on_origin=True,
                     colors=grid_colors, box_height=1, grid_label_templates=grid_label_templates)

    ax1.set_ylim(-1, 2)
    ax1.set_axis_on()
    ax1.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        labelbottom=False,
        labelleft=False)
    ax1.set_title("Granulation of Space")

    # number of points
    num_points = len(x_point_vals)

    for j in range(1, num_points):
        origin_point = (x_point_vals[j - 1], 0.0)
        width = x_point_vals[j] - x_point_vals[j - 1]
        height = 1

        # create rectangle with text inside
        add_text_rect(ax1, origin_point[0], origin_point[1], width, height, angle=0, linewidth=1.5, edgecolor='k',
                      fontsize=8, facecolor='none', text_str="%.2f" % width, alpha=1.0,
                      text_v_offset=min(-0.1, -height / 1.2))

    ax2.vlines(boundary_x, ymin=0, ymax=grid_delta_counts, color='k')
    ax2.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True)
    ax2.yaxis.set_major_locator(ticker.IndexLocator(1, 0))

    ax2.set_ylim(bottom=0)
    ax2.set_title("Change-Point Count")
    ax2.set_xlabel("Unit Interval")

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')


if __name__ == "__main__":

    lower_k = 2
    upper_k = 9
    # number of bins per partition
    partition_sizes = list(range(lower_k, upper_k + 1))
    # partition_sizes = [2, 3, 5, 7, 11]
    # partition_sizes = [2,3,5] #,5,7]

    # partition_sizes = [2, 4, 7, 11]
    # partition_sizes = [2, 3, 4, 6, 8]
    # partition_sizes = list(range(2, 10))
    # partition_sizes = list(range(2, 5))
    # prime_partition_sizes = [2, 3, 5, 7, 11, 13, 17, 23]
    # partition_sizes = prime_partition_sizes

    # for n_grids in range(1, 5):
    # for n_grids in range(8, 9):
    # for n_grids in range(1, 2):
    # for n_grids in range(2, 3):
    # for n_grids in range(1, 4):
    # for n_grids in range(3, 4):
    # for n_grids in range(1, 9):
    # for n_grids in range(8, 9):
    for n_grids in range(1, 9):

        print(n_grids)
        print(partition_sizes)

        k_comb = list(it.combinations(partition_sizes, n_grids))

        for parts in k_comb:

            # create filename
            file_name = "out/bin_interval_graph_%02u_grids_bins" % n_grids
            for k in parts:
                file_name += "_%02u" % k
            file_name += ".png"

            print(file_name)
            plot_interval_graph_1(parts, file_name=file_name)
            # plot_interval_graph_2(parts, file_name=file_name)
            # plot_interval_graph_3(parts, file_name=file_name)
            plt.clf()
