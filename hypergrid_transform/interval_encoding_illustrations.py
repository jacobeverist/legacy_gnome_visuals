from intervals import FloatInterval as I
from intervals.exc import *
import numpy as np
import pprint
import itertools as it
import string
from fractions import Fraction

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches


# Analysis of finite interval encoders
# assuming only over the unit interval and all grid periods = 1

def gnome_similarity(X_gnomes, ref_gnomes):
    """Gnome similarity score.

    Asymmetric score with respect to X_gnomes

    :param X_gnomes: array_like
        An array with shape (n_samples, n_features).

    :param ref_gnomes: array_like
        An array with shape (n_samples, n_features).

    :return: array
        An array of scores from 0.0 to 1.0 with shape (n_samples)

    """

    # FIXME: Normalize based on ref_gnomes bit count, not X_gnomes

    sum_scores = np.dot(X_gnomes.astype(int), ref_gnomes.T.astype(int))
    l1_norm = np.count_nonzero(X_gnomes, axis=1)

    # replace any 0 values with 1
    l1_norm[l1_norm == 0] = 1

    normalized_scores = np.divide(sum_scores, l1_norm[:, np.newaxis])

    return normalized_scores


def place_similarity(X_one_shot, ref_one_shot):
    """ Place distance of one-shot representations

    """

    # verify that inputs are one-shot
    one_count = np.count_nonzero(X_one_shot, axis=1)
    if np.any(one_count != 1):
        raise Exception("an input is not one-shot binary")

    one_count = np.count_nonzero(ref_one_shot, axis=1)
    if np.any(one_count != 1):
        raise Exception("an input is not one-shot binary")

    # number of bits from the shape
    n_bits = X_one_shot.shape[1]

    X_index = np.argmax(X_one_shot == 1, axis=1)
    ref_index = np.argmax(ref_one_shot == 1, axis=1)

    diff_scores = np.abs(np.subtract(X_index.reshape(-1, 1).astype(int), ref_index.reshape(1, -1).astype(int)))

    similarity_scores = n_bits - diff_scores

    normalized_scores = np.divide(similarity_scores, n_bits)

    return normalized_scores

def draw_encoding(ax, X_gnomes):
    # print("encoding:")
    # print(X_gnomes.shape)

    state_data = np.rot90(X_gnomes, k=-1, axes=(1, 0))
    # print(state_data.shape)

    # for k in range(X_gnomes.shape[0]):
    #    print(X_gnomes[k,:])

    # for k in range(state_data.shape[1]):
    #    print(state_data[:,k])

    barprops = dict(cmap='binary', interpolation='nearest', aspect='auto',
                    extent=[0, state_data.shape[1], 0, state_data.shape[0]])
    # barprops = dict(cmap='binary')
    # extent=[0, state_data.shape[1], 0, state_data.shape[0]])
    img = ax.imshow(state_data, **barprops)


def draw_delta_count(ax, boundary_x, grid_delta_counts):
    ax.vlines(boundary_x, ymin=0, ymax=grid_delta_counts, color='k')

    tick_points_x = []
    tick_points_y = []

    for k in range(len(grid_delta_counts)):
        count = grid_delta_counts[k]
        x_val = boundary_x[k]

        tick_points_x += [x_val for j in range(count)]
        tick_points_y += [j + 1 for j in range(count)]

    # ax3.scatter(boundary_x, grid_delta_counts, color='k')
    ax.scatter(tick_points_x, tick_points_y, color='k')


def draw_granulation(ax, boundaries):
    x_point_vals = []
    y_point_vals = []
    for j in range(0, len(boundaries)):
        x_point_vals.append(boundaries[j])
        y_point_vals.append(boundaries[j])

    # number of points
    num_points = len(x_point_vals)

    frac_sum = Fraction(0)

    for j in range(1, num_points):
        origin_point = (x_point_vals[j - 1], 0.0)
        width = x_point_vals[j] - x_point_vals[j - 1]
        height = 1

        # text_v_offset = min(-0.1, -height / 1.2)
        # text_v_offset = min(-0.1, -height*0.55)
        # text_v_offset = 0.0

        frac = Fraction(str(width)).limit_denominator(1000)
        # print("%s" % str(frac))
        frac_sum = frac_sum + frac
        text_str = "%d\n--\n%d" % (frac.numerator, frac.denominator)
        # text_str += "\n\n\n%.2f" % width
        # text_str = "%d\n\u2014\n%d" % (frac.numerator, frac.denominator)

        # create rectangle with text inside
        # add_text_rect(ax1, origin_point[0], origin_point[1], width, height, angle=0, linewidth=1.5, edgecolor='k',
        #              fontsize=8, facecolor='none', text_str="%.2f" % width, alpha=1.0,
        #              text_v_offset=text_v_offset)

        add_text_rect(ax, origin_point[0], origin_point[1], width, height, angle=0, linewidth=1.5, edgecolor='k',
                      fontsize=8, facecolor='none', alpha=1.0, text_str=text_str)

    print("total sum = %s" % str(frac_sum))


def draw_encoder_bins(ax, encoders, input_range=(0, 1), clip_on=True, spacing=1):
    # def add_text_rect(ax, box_x, box_y, box_width, box_height, angle=0, linewidth=1.5, edgecolor='k', fontsize=8,
    #              facecolor='none', text_str=None, aligned_text=False, alpha=1.0, text_v_offset=-0.01):

    min_y = 1
    max_y = 0

    keys = list(encoders.keys())
    keys.sort()

    n_grids = len(keys)
    grid_names = string.ascii_uppercase[:n_grids]

    colors = sns.color_palette("muted", n_colors=n_grids)
    grid_label_templates = [prefix + "%d" for prefix in grid_names]
    grid_colors = [colors[j] for j in range(n_grids)]

    grid_labels = ["%d,%d" % (keys[j], spacing) for j in range(n_grids)]

    # add_1D_scalar_encoder(ax0, bins=partition_sizes, periods=dist_periods, minimal_range=True, boundary_on_origin=True,
    #                      colors=grid_colors, box_height=1, grid_label_templates=grid_label_templates)

    #             else:
    #                 add_text_rect(ax, corner_pos[0], corner_pos[1], bin_interval_x, box_height, angle=angle,
    #                               text_str=grid_label_templates[grid_i] % (interval_count % n_bins,),
    #                               aligned_text=aligned_text, facecolor=total_color,
    #                               alpha=alpha, linewidth=linewidth)

    encoder_count = 0
    bin_id_count = 0
    draw_y = 0.0
    # box_height = 0.3
    box_height = 1

    top_right_points_x = []
    top_right_points_y = []

    for key in keys:
        encoder = encoders[key]

        bin_count = 0

        for bin in encoder:
            upper_bound = bin.upper
            lower_bound = bin.lower

            box_x = lower_bound
            box_y = draw_y
            box_width = upper_bound - lower_bound

            if clip_on:
                # case 1: bin exceeds lower bound
                if box_x < input_range[0]:
                    box_x = input_range[0]
                    box_width = upper_bound - box_x

                # case 2: bin exceeds interval upper bound
                elif box_x + box_width > input_range[1]:
                    box_width = input_range[1] - box_x

                # case 3: bin within interval bounds
                else:
                    # do nothing
                    pass

            if bin_count == 0:
                add_text_rect(ax, box_x, box_y, box_width, box_height, alpha=0.8, facecolor=grid_colors[encoder_count],
                              text_str=str(bin_id_count), clip_on=clip_on, linewidth=0.2,
                              label=grid_labels[encoder_count])
            else:
                add_text_rect(ax, box_x, box_y, box_width, box_height, alpha=0.8, facecolor=grid_colors[encoder_count],
                              text_str=str(bin_id_count), clip_on=clip_on, linewidth=0.2)
            # text_str=grid_label_templates[encoder_count] % bin_count, clip_on=clip_on, linewidth=0.2)

            bin_count += 1
            bin_id_count += 1
            draw_y += box_height

            if box_y < min_y:
                min_y = box_y

            if box_y + box_height > max_y:
                max_y = box_y + box_height

            # top_right_points_x.append(box_x+box_width)
            # top_right_points_y.append(box_y+box_height)

        # draw_y += box_height
        encoder_count += 1
        draw_y = box_height * encoder_count * spacing

    # ax.scatter(top_right_points_x, top_right_points_y, s=0.01)

    return max_y, min_y


def add_text_rect(ax, box_x, box_y, box_width, box_height, angle=0, linewidth=1.5, edgecolor='k', fontsize=8,
                  facecolor='none', text_str=None, aligned_text=False, alpha=1.0, text_v_offset=-0.01, clip_on=False,
                  label=None):
    # print("add_text_rect(args):")
    # print("box_x:", box_x)
    # print("box_y:", box_y)
    # print("box_width:", box_width)
    # print("box_height:", box_height)
    # print("angle:", angle)

    # data space coordinates to find new point after rotated
    fixed_point_rotation = Affine2D().rotate_deg_around(box_x, box_y, angle)

    # add rectangle at corner position and rotate by angle
    rect = patches.Rectangle((box_x, box_y), box_width, box_height, angle=angle, linewidth=linewidth,
                             edgecolor=edgecolor,
                             facecolor=facecolor, clip_on=clip_on, alpha=alpha, label=label)
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
            fontsize=fontsize, va='center', ha='center', clip_on=clip_on, alpha=alpha)


def create_n_overlaps_of_w(n, w, interval_to_partition, is_debug=False):
    if is_debug:
        print("%d bins, w=%d" % (n, w))
        print("------------------")

    upper_bound = interval_to_partition.upper
    lower_bound = interval_to_partition.lower
    interval_size = upper_bound - lower_bound

    # INVARIANT
    # 2*w <= n
    if 2 * w > n:
        raise Exception("n=%d,w=%d does not satisfy condition 2*w <= n" % (n, w))

    # step size
    # 1 / (n-w+1)
    num_partitions = n - (w - 1)
    step_size = 1 / num_partitions * interval_size
    # step_size = (1 / n) * interval_size
    # num_steps = n

    bin_size = step_size * w
    if is_debug:
        print("%.2f interval size" % interval_size)
        print("%d bins" % n)
        print("%.2f step_size" % step_size)
        print("%.2f bin_size" % bin_size)
        print("------------------")

    equidist_points = np.linspace(lower_bound, upper_bound, endpoint=True, num=num_partitions + 1)
    # equidist_points = np.linspace(lower_bound, upper_bound, endpoint=True, num=n + 1)

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

    # intervals = [I.closedopen(val_list[c - 1], val_list[c]) for c in range(1, len(val_list))]
    if is_debug:
        print("partition:", intervals)

    return intervals


def iter_all_w_combinations(partition_dict):
    partition_keys = list(partition_dict.keys())
    partition_keys.sort()

    print("partition_keys:", partition_keys)
    print("partition_dict:")
    for k in partition_keys:
        print(partition_dict[k])

    combo_intervals = []
    for k in partition_keys:
        bin_intervals = partition_dict[k]
        combo_intervals.append(bin_intervals)

    print("combo_intervals:")
    for k in range(len(combo_intervals)):
        print(combo_intervals[k])

    prod_combos = list(it.product(*combo_intervals))
    print("prod_combos:")
    for k in range(len(prod_combos)):
        print(prod_combos[k])
    print()

    return prod_combos


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

        print("overlapped:", overlapped_interval)

        # return quickly if its already empty
        if overlapped_interval.empty:
            return I.from_string('(0, 0]')
            #return I.empty()

    return overlapped_interval


def find_grid_self_intersections(grid_bins, w=1):
    if w < 1:
        raise Exception("w must be greater than 1")

    if w > len(grid_bins):
        raise Exception("w must be less than n")

    # take every w set of bins as a self-intersection
    grid_intersections = []
    for i in range(len(grid_bins) - w + 1):
        grid_intersections.append(tuple([i + j for j in range(w)]))

    # print("grid intersections")
    # print(grid_intersections)

    interval_combinations = []

    for comb in grid_intersections:
        intervals = []

        # comb is a tuple of bin indices for a grid self-intersection
        for i in comb:
            bin_interval = grid_bins[i]
            intervals.append(bin_interval)

        intersected_interval = intersect_intervals(intervals)

        if not intersected_interval.empty:
            interval_size = intersected_interval.upper - intersected_interval.lower

            interval_combinations.append((interval_size, intersected_interval, intervals, comb))

    return interval_combinations


def find_self_intersected_combinations(encoder_granulations):

    # interval_combinations.append((interval_size, intersected_interval, intervals, comb))

    interval_combinations = []

    from datetime import datetime

    # print("combinations")
    count = 0

    t0 = datetime.now()

    """
    def iter_all_w_combinations(partition_dict):
    
        partition_keys = list(partition_dict.keys())
        partition_keys.sort()

        combo_intervals = []
        for k in partition_keys:
            bin_intervals = partition_dict[k]
            combo_intervals.append(bin_intervals)

        return it.product(*combo_intervals)
        
    def intersect_intervals(interval_list):
    
        if not len(interval_list) > 0:
            return I.empty()

        overlapped_interval = interval_list[0]
        for interval in interval_list[1:]:
            overlapped_interval = overlapped_interval & interval

            # return quickly if its already empty
            if overlapped_interval.empty:
                return I.empty()

        return overlapped_interval
    """

    granulation_keys = list(encoder_granulations.keys())
    granulation_keys.sort(reverse=True)

    print("encoder_granulations:")
    print(encoder_granulations)
    print("granulation_keys:")
    print(granulation_keys)

    num_encoders = len(granulation_keys)

    # granulation interval combinations indices
    combo_index = [0 for i in range(num_encoders)]

    # encoder granulations: dict
    # { n : [(bin_ids, interval), ...],  }

    # TODO:  optimize this loop so it only explores combinations that are self-intersecting

    """
    while True:

        bin_id, intermediate_interval = encoder_granulations[granulation_keys[0]][combo_index[0]]

        next_index = 1
        is_non_empty = True

        while is_non_empty and next_index < num_encoders:
            next_bin_id, next_interval = encoder_granulations[granulation_keys[next_index]][combo_index[next_index]]
            intermediate_interval = intermediate_interval & next_interval

            is_non_empty = not intermediate_interval.empty

            next_index += 1

        # successful granulation combo found
        if is_non_empty and next_index == num_encoders:
            print("found:", combo_index)
            #interval_size = intermediate_interval.upper - intermediate_interval.lower

            #interval_combinations.append((interval_size, intermediate_interval, intervals, tuple(bins)))

        # iterate through product of combinations
        is_done = False
        encoder_index = num_encoders - 1
        while not is_done:

            # increment
            combo_index[encoder_index] += 1

            # if exceeds number of granulated intervals
            # carry over to next encoder increment
            if combo_index[encoder_index] >= len(encoder_granulations[granulation_keys[encoder_index]]):

                combo_index[encoder_index] = 0
                encoder_index -= 1
            else:
                is_done = True

            if encoder_index < 0:
                is_done = True

        # break loop if we've covered all combinations
        if encoder_index < 0:
            break
    """

    for comb in iter_all_w_combinations(encoder_granulations):

        intervals = []
        bins = []
        print("comb:", comb)
        # FIXME: changed this because it seemed it was expecting
        # FIXME: a different data structure
        #for bin, overlap_intervals in comb:
        for bin, overlap_interval in comb:

            print("bin, overlap_interval:", bin, overlap_interval)

            # FIXME: changed this because it seemed it was expecting
            # FIXME: a different data structure
            #for interval in overlap_intervals:
            #    intervals.append(interval)
            #bins.append(bin)
            intervals.append(overlap_interval)
            bins.append(bin)

        intersected_interval = intersect_intervals(intervals)

        if not intersected_interval.empty:
            print("non-empty", count)
            interval_size = intersected_interval.upper - intersected_interval.lower

            interval_combinations.append((interval_size, intersected_interval, intervals, tuple(bins)))

        count += 1

    t1 = datetime.now()
    # print("total time:", t1-t0)

    return interval_combinations


def encode_binary_integer(ref_vals, n_bits, x_min=0.0, x_max=1.0):
    interval_to_partition = I.closed(x_min, x_max)

    n_bins = 2 ** n_bits

    bin_list = create_n_overlaps_of_w(n_bins, 1, interval_to_partition, is_debug=False)

    gnomes = []
    for x in ref_vals:
        bin_index = None
        for k in range(len(bin_list)):
            if x in bin_list[k]:
                bin_index = k

        if bin_list is None:
            raise Exception("Referenced value %f is not within range [%f,%f]" % (x, x_min, x_max))

        #print("%f => %d => %s => %s" % (x, bin_index, format(bin_index, "0%db" % n_bits),
        #                                str(np.array([int(k) for k in format(bin_index, "0%db" % n_bits)]))))
        gnomes.append(np.array([int(k) for k in format(bin_index, "0%db" % n_bits)]))

    # reverse bit order so it aligns with base-2 encoding places
    gnome_arrays = np.array(gnomes)
    gnome_arrays = np.flip(gnome_arrays, axis=1)

    print("n_bits=", n_bits)
    print("n_bins=", n_bins)
    print(len(bin_list), "bins")
    return gnome_arrays
    #return np.array(gnomes)


def encode_one_hot(ref_vals, n_bits, x_min=0.0, x_max=1.0):
    interval_to_partition = I.closed(x_min, x_max)

    bin_list = create_n_overlaps_of_w(n_bits, 1, interval_to_partition)

    gnomes = []
    for x in ref_vals:
        bin_index = None
        for k in range(len(bin_list)):
            if x in bin_list[k]:
                bin_index = k

        if bin_list is None:
            raise Exception("Referenced value %f is not within range [%f,%f]" % (x, x_min, x_max))

        # print("%f => %d => %s" % (x, bin_index, format(bin_index, "0%db" % n_bits)))
        gnomes.append(np.array([1 if x in interval else 0 for interval in bin_list]))

    return np.array(gnomes)


def encode_with_bins(ref_vals, bin_list):
    gnomes = []
    for x in ref_vals:
        gnomes.append(np.array([1 if x in interval else 0 for interval in bin_list]))

    return np.array(gnomes)


def plot_interval_graph_3(encoder_params=((2, 1),), lower_bound=0, upper_bound=1, file_name=None):
    unit_interval = I.closed(lower_bound, upper_bound)

    k_partitions = {}
    k_grid_self_intersections = {}
    # print("encoder_params:", encoder_params)

    # 2 independent, 1 dependent
    # n = number of bins
    # w = number of adjacent bins that overlap
    # s = step size, k = partitions, k = 1/s

    # specify k and w
    # generate n bins, starting at [(1-w)*s+i*s, (i+1)*s], for i'th bin

    # create identically sized partition of the unit interval into k components
    # k is the number of bins
    # w, s.t. w < k, where w is number of bins containing any point in interval
    all_bins = []
    for n, w in encoder_params:
        k_partitions[n] = create_n_overlaps_of_w(n, w, unit_interval, is_debug=False)

        all_bins += k_partitions[n]

        # print()
        # print("grid bins, configured/actual: %d, %d" % (n, len(k_partitions[n])))
        # pp = pprint.PrettyPrinter()
        # pp.pprint(k_partitions[n])
        # print()

        # 1) find self-intersections for a single grid for w > 1
        #

        grid_interval_combinations = find_grid_self_intersections(k_partitions[n], w=w)
        # print("%d total grid self-intersections" % len(grid_interval_combinations))
        #  grid_interval_combinations.append((interval_size, intersected_interval, intervals, comb))

        new_intervals = []

        # print("(size, intersection, component bins, bin_ids)")
        for comb in grid_interval_combinations:
            new_intervals.append((comb[3], comb[1]))
        #    print(comb)

        k_grid_self_intersections[n] = new_intervals

        # print("(bin_ids, intersection)")
        # for comb in new_intervals:
        #    print(comb)

    print("k_grid_self_intersections:")
    for n, w in encoder_params:
        print(k_grid_self_intersections[n])

    # reference points for comparison
    ref_points = np.array([[0.21], [0.7]])

    # conversions
    ref_gnomes = encode_with_bins(ref_points, all_bins)
    ref_one_hot = encode_one_hot(ref_points, 6)
    ref_binary_integers = encode_binary_integer(ref_points, 6, x_min=0.0, x_max=1.0)

    # print("ref_points:", ref_points.shape)
    # print(ref_points)
    # print("ref_gnomes:", ref_gnomes.shape)
    # print(ref_gnomes)
    # print(ref_binary_integers)

    X_points = np.arange(0, 1, 0.001).reshape(-1, 1)

    # conversions
    X_gnomes = encode_with_bins(X_points, all_bins)
    X_one_hot = encode_one_hot(X_points, 6)
    X_binary_integers = encode_binary_integer(X_points, 6, x_min=0.0, x_max=1.0)

    # print(X_points.shape)
    # print(X_gnomes.shape)

    # print(X_binary_integers)

    scores = gnome_similarity(X_gnomes, ref_gnomes)
    binary_scores = gnome_similarity(X_binary_integers, ref_binary_integers)
    place_scores = place_similarity(X_one_hot, ref_one_hot)
    # print(scores)
    # print(binary_scores)
    # print(place_scores)

    # 2) find combinations of self-intersections across grids
    # 3) compute which combinations are intersecting

    interval_combinations = find_self_intersected_combinations(k_grid_self_intersections)
    print()
    print("%d total combinations" % len(interval_combinations))
    print("(size, intersection, component bins, bin_ids)")
    for comb in interval_combinations:
       print(comb)

    # number of grids is the number of partition sizes
    n_grids = len(encoder_params)
    # grid_names = string.ascii_uppercase[:n_grids]

    # total number of bins across all grids
    total_n_bins = 0
    for n, w in encoder_params:
        total_n_bins += len(k_partitions[n])

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

    # colors = sns.color_palette("muted", n_colors=n_grids)

    boundary_x = x_point_vals[1:-1]

    """
    boundary_y = y_point_vals[1:-1]
    boundary_colors = []
    for k in range(len(grid_deltas)):
        boundary_colors.append(colors[grid_deltas[k]])
    """

    """
    all_labels = [str(encoder_params[j]) + " intervals" for j in range(len(encoder_params))]
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
    """

    n_encoders = n_grids
    n_bins_last_encoder = encoder_params[-1][0]
    y_max = n_grids + n_bins_last_encoder

    # FIXME: Assumes W is same for all encoders
    w_size = encoder_params[0][1]

    # grid_label_templates = [prefix + "%d" for prefix in grid_names]
    # dist_periods = [1.0 for j in range(n_grids)]
    # grid_colors = [colors[j] for j in range(n_grids)]

    # Draw Plots in Each SubAxes

    fig = plt.figure(num=1, figsize=(10, 8), constrained_layout=True)
    # fig, axes = plt.subplots(3, 1, num=1, sharex=True)
    fig, axes = plt.subplots(5, 1, num=1)  # , sharex=True)

    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    ax3 = axes[3]
    ax4 = axes[4]

    ax0.get_shared_x_axes().join(ax0, ax1)
    ax0.get_shared_x_axes().join(ax0, ax2)
    ax0.get_shared_x_axes().join(ax0, ax4)

    title_str = "Finite Interval = [0,1]\n"
    title_str += "Encoders (n,w) = "
    for j in range(len(encoder_params) - 1):
        title_str += "(%d,%d), " % encoder_params[j]
    title_str += "(%d,%d)" % encoder_params[-1]

    title_str = "Finite Interval Encoding"
    fig.suptitle(title_str)

    # Encoder Bins

    max_y, min_y = draw_encoder_bins(ax0, k_partitions, spacing=w_size)

    ax0.set_xlim(0.0, 1.0)
    ax0.set_ylim(min_y - 1, max_y + 1)
    ax0.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=False,
        labelbottom=False,
        labelleft=False)
    ax0.set_ylabel("Encoder Bins")
    ax0.vlines([0.0, 1.0], ymin=-100, ymax=100, color='k', alpha=0.2)

    handles, labels = ax0.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    # ax0.legend(handles, labels, fontsize=8, loc='upper left')
    # fig.legend(handles, labels, fontsize=8, bbox_to_anchor=(1.05, 0.9))
    # fig.legend(handles, labels, fontsize=8, bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
    # fig.legend(handles, labels, title="Encoders (n,w)", fontsize=8, title_fontsize=8, bbox_to_anchor=(1.06, 1), bbox_transform=ax0.transAxes)
    fig.legend(handles, labels, title="n,w", fontsize=8, title_fontsize=8, bbox_to_anchor=(1.08, 1),
               bbox_transform=ax0.transAxes)

    # Granulation

    draw_granulation(ax1, boundaries)

    ax1.set_ylim(0.0, 1.0)
    ax1.set_axis_on()
    ax1.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=False,
        labelbottom=False,
        labelleft=False)
    ax1.set_ylabel("Granulation")
    ax1.vlines([0.0, 1.0], ymin=-100, ymax=100, color='k', alpha=0.2)

    # Delta Count

    draw_delta_count(ax2, boundary_x, grid_delta_counts)

    max_change_count = max(grid_delta_counts)

    ax2.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=False,
        right=True,
        labelbottom=False,
        labelleft=False,
        labelright=True)

    ax2.set_ylim(bottom=0, top=max_change_count + 1)
    ax2.set_ylabel("Delta\nCount")
    ax2.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    # ax3.set_ylim(bottom=0, top=max_change_count)

    # Similarity Scores

    draw_encoding(ax3, X_gnomes)

    ax3.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        right=True,
        labelbottom=False,
        labelleft=False,
        labelright=True)
    ax3.set_ylabel("Encoded\nOutput")

    ax3.set_ylim(0, total_n_bins)

    #   deep, muted, bright, pastel, dark, colorblind
    # colors = sns.color_palette("Set1", desat=.5, n_colors=len(ref_points))
    colors = sns.color_palette("Set1", n_colors=len(ref_points))
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    for k in range(len(ref_points)):
        # ax4.plot(X_points, scores[:, k], color=colors[k], label=float(ref_points[k]))
        # ax4.plot(X_points, binary_scores[:, k], color=colors[k], label=float(ref_points[k]))
        ax4.plot(X_points, place_scores[:, k], color=colors[k], label=float(ref_points[k]))
        ax4.axvline(x=ref_points[k], ymax=5.7, alpha=0.4, linewidth=1.5, color=colors[k], clip_on=False)

    # print(X_points.shape)
    # print(scores.shape)
    # print(ref_points.shape)

    # ax4.plot(X_points, scores, label=ref_points.transpose())

    ax4.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=False,
        right=True,
        labelbottom=True,
        labelleft=False,
        labelright=True)
    ax4.set_ylabel("Similarity\nComparison")
    # ax4.set_xlabel("Unit Interval")
    ax4.yaxis.set_major_locator(ticker.IndexLocator(1, 0))

    ax4.set_xlabel("Input Range [0,1]")

    # create the legend on the figure, not the axes
    handles, labels = ax4.get_legend_handles_labels()
    # ax4.legend(handles, labels, title="Input\nComparisons", title_fontsize=8, fontsize=8)
    ax4.legend(handles, labels, fontsize=8)
    # fig.legend(handles, labels, title="Comparison\nPoints", loc='upper right',
    #              bbox_to_anchor=(1.1, 0.9))  # , fontsize=20)

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')


def plot_similarity_1(n_bits=16, lower_bound=0, upper_bound=1, file_dir="./"):
    file_name = file_dir + "%02u_base_2_similarity" % n_bits
    file_name += ".png"

    # reference points for comparison
    # ref_points = np.array([[0.21], [0.7]])
    # ref_points = np.array([[0.51]])
    ref_points = np.array([[0.61]])

    # conversions
    # ref_gnomes = encode_with_bins(ref_points, all_bins)
    ref_one_hot = encode_one_hot(ref_points, n_bits)
    ref_binary_integers = encode_binary_integer(ref_points, n_bits, x_min=lower_bound, x_max=upper_bound)

    # X_points = np.arange(lower_bound, upper_bound, 0.001).reshape(-1, 1)
    X_points = np.arange(lower_bound, upper_bound, 0.001).reshape(-1, 1)

    # conversions
    # X_gnomes = encode_with_bins(X_points, all_bins)
    X_one_hot = encode_one_hot(X_points, n_bits)
    X_binary_integers = encode_binary_integer(X_points, n_bits, x_min=lower_bound, x_max=upper_bound)

    # scores = gnome_similarity(X_gnomes, ref_gnomes)
    binary_scores = gnome_similarity(X_binary_integers, ref_binary_integers)
    place_scores = place_similarity(X_one_hot, ref_one_hot)

    # Draw Plots in Each SubAxes

    fig = plt.figure(num=1, figsize=(10, 8), constrained_layout=True)
    # fig, axes = plt.subplots(5, 1, num=1)  # , sharex=True)
    fig, axes = plt.subplots(2, 1, num=1)  # , sharex=True)

    ax0 = axes[0]
    ax1 = axes[1]
    # ax2 = axes[2]
    # ax3 = axes[3]
    # ax4 = axes[4]
    # ax0.get_shared_x_axes().join(ax0, ax1)
    # ax0.get_shared_x_axes().join(ax0, ax2)
    # ax0.get_shared_x_axes().join(ax0, ax4)

    # ax4 = axes
    ax3 = ax0
    ax4 = ax1

    title_str = "Base-2 Integer Encoding"
    title_str += "\nBits = %d" % n_bits
    title_str += "\nEncoded Value = %0.2f" % ref_points[0]
    fig.suptitle(title_str)

    draw_encoding(ax3, X_binary_integers)
    ax3.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        right=True,
        labelbottom=False,
        labelleft=False,
        labelright=True)
    ax3.set_ylabel("Encoded\nOutput")
    ax3.set_ylim(0, n_bits)

    # palettes:  deep, muted, bright, pastel, dark, colorblind
    # colors = sns.color_palette("Set1", n_colors=len(ref_points))
    colors = sns.color_palette("Set1", n_colors=3)
    ax4.axvline(x=ref_points[0], ymax=2.1, alpha=0.4, linewidth=1.5, color=colors[0], clip_on=False)

    ax4.plot(X_points, binary_scores[:, 0], color=colors[1], label="Bit Similarity")
    # ax4.plot(X_points, place_scores[:, 0], color=colors[2], label="One-Hot Place Distance")
    # ax4.plot(X_points, scores[:, 0], color=colors[2], label="Scalar Encoder Similarity")

    # for k in range(len(ref_points)):
    #    ax4.plot(X_points, place_scores[:, k], color=colors[k], label=float(ref_points[k]))
    #    ax4.axvline(x=ref_points[k], ymax=5.7, alpha=0.4, linewidth=1.5, color=colors[k], clip_on=False)

    ax4.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=False,
        right=True,
        labelbottom=True,
        labelleft=False,
        labelright=True)
    ax4.set_ylabel("Bit Similarity")
    ax4.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax4.set_xlim(lower_bound, upper_bound)

    ax4.set_xlabel("Input Range [0,1]")

    # create the legend on the figure, not the axes
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels, fontsize=8)

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')


def plot_encoding_1(n_bits=16, lower_bound=0, upper_bound=1, file_dir="./"):
    file_name = file_dir + "%02u_encoding" % n_bits
    file_name += ".png"

    unit_interval = I.closed(lower_bound, upper_bound)
    w = 3
    all_bins = create_n_overlaps_of_w(n_bits, w, unit_interval, is_debug=False)

    # reference points for comparison
    # ref_points = np.array([[0.21], [0.7]])
    # ref_points = np.array([[0.51]])
    ref_points = np.array([[0.61]])

    # conversions
    ref_gnomes = encode_with_bins(ref_points, all_bins)
    ref_one_hot = encode_one_hot(ref_points, n_bits)
    ref_binary_integers = encode_binary_integer(ref_points, n_bits, x_min=lower_bound, x_max=upper_bound)

    X_points = np.arange(lower_bound, upper_bound, 0.001).reshape(-1, 1)

    # conversions
    X_gnomes = encode_with_bins(X_points, all_bins)
    X_one_hot = encode_one_hot(X_points, n_bits)
    X_binary_integers = encode_binary_integer(X_points, n_bits, x_min=lower_bound, x_max=upper_bound)

    scores = gnome_similarity(X_gnomes, ref_gnomes)
    binary_scores = gnome_similarity(X_binary_integers, ref_binary_integers)
    place_scores = place_similarity(X_one_hot, ref_one_hot)

    # Draw Plots in Each SubAxes

    fig = plt.figure(num=1, figsize=(10, 8), constrained_layout=True)
    # fig, axes = plt.subplots(5, 1, num=1)  # , sharex=True)
    fig, axes = plt.subplots(3, 1, num=1)  # , sharex=True)

    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]

    #title_str = "Encodings on Interval [0,1]"
    #fig.suptitle(title_str)

    draw_encoding(ax0, X_binary_integers)
    ax0.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=True,
        right=False,
        labelbottom=False,
        labelleft=True,
        labelright=False)
    # ax0.set_ylabel("Base-2 Integer Encoding")
    ax0.set_ylim(0, n_bits)
    ax0.set_title("Base-2 Integer Encoding, n=%d" % n_bits)
    ax0.yaxis.set_major_locator(ticker.IndexLocator(1, 0))

    encoder_shape = X_binary_integers.shape
    line_indices = [k for k in range(1, encoder_shape[1])]
    ax0.hlines(line_indices, 0, encoder_shape[0], alpha=0.2)

    draw_encoding(ax1, X_one_hot)
    ax1.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=True,
        right=False,
        labelbottom=False,
        labelleft=True,
        labelright=False)
    # ax1.set_ylabel("One-Hot Encoding")
    ax1.set_ylim(0, n_bits)
    ax1.set_title("One-Hot Encoding, n=%d" % n_bits)
    ax1.yaxis.set_major_locator(ticker.IndexLocator(1, 0))

    encoder_shape = X_one_hot.shape
    line_indices = [k for k in range(1, encoder_shape[1])]
    ax1.hlines(line_indices, 0, encoder_shape[0], alpha=0.2)

    draw_encoding(ax2, X_gnomes)
    ax2.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        right=False,
        labelbottom=True,
        labelleft=True,
        labelright=False)
    # ax2.set_ylabel("Scalar Encoding, w=%d" % w)
    ax2.set_ylim(0, n_bits)
    ax2.set_title("Scalar Encoding, n=%d, w=%d" % (n_bits, w))

    ax2.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    # ax2.xaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax2.xaxis.set_major_locator(ticker.LinearLocator(5))
    # labels = [item.get_text() for item in ax2.get_xticklabels()]
    # labels[1] = 'Testing'
    labels = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    ax2.set_xticklabels(labels)

    encoder_shape = X_gnomes.shape
    line_indices = [k for k in range(1, encoder_shape[1])]
    ax2.hlines(line_indices, 0, encoder_shape[0], alpha=0.2)

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')


def plot_similarity_2(n_bits=16, lower_bound=0, upper_bound=1, file_dir="./"):
    file_name = file_dir + "%02u_similarity" % n_bits
    file_name += ".png"

    unit_interval = I.closed(lower_bound, upper_bound)
    w = 3
    all_bins = create_n_overlaps_of_w(n_bits, w, unit_interval, is_debug=False)

    # reference points for comparison
    # ref_points = np.array([[0.21], [0.7]])
    # ref_points = np.array([[0.51]])
    ref_points = np.array([[0.21], [0.69]])
    #ref_points = np.array([[0.11], [0.31], [0.61]])

    # conversions
    ref_gnomes = encode_with_bins(ref_points, all_bins)
    ref_one_hot = encode_one_hot(ref_points, n_bits)
    ref_binary_integers = encode_binary_integer(ref_points, n_bits, x_min=lower_bound, x_max=upper_bound)

    X_points = np.arange(lower_bound, upper_bound, 0.001).reshape(-1, 1)

    # conversions
    X_gnomes = encode_with_bins(X_points, all_bins)
    X_one_hot = encode_one_hot(X_points, n_bits)
    X_binary_integers = encode_binary_integer(X_points, n_bits, x_min=lower_bound, x_max=upper_bound)

    scores = gnome_similarity(X_gnomes, ref_gnomes)
    binary_scores = gnome_similarity(X_binary_integers, ref_binary_integers)
    #place_scores = place_similarity(X_one_hot, ref_one_hot)
    one_hot_scores = gnome_similarity(X_one_hot, ref_one_hot)

    # Draw Plots in Each SubAxes

    fig = plt.figure(num=1, figsize=(10, 8), constrained_layout=True)
    # fig, axes = plt.subplots(5, 1, num=1)  # , sharex=True)
    fig, axes = plt.subplots(3, 1, num=1)  # , sharex=True)

    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]

    colors = sns.color_palette("Set1", n_colors=len(ref_points))

    #title_str = "Encodings on Interval [0,1]"
    #fig.suptitle(title_str)

    ax0.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=True,
        right=False,
        labelbottom=False,
        labelleft=True,
        labelright=False)
    ax0.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax0.set_title("Base-2 Integer Bit Similarity, n=%d" % n_bits)
    for k in range(len(ref_points)):
        ax0.plot(X_points, binary_scores[:, k], color=colors[k], label=float(ref_points[k]))

    ax1.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=True,
        right=False,
        labelbottom=False,
        labelleft=True,
        labelright=False)
    ax1.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    for k in range(len(ref_points)):
        ax1.plot(X_points, one_hot_scores[:, k], color=colors[k], label=float(ref_points[k]))

    ax1.set_title("One-Hot Bit Similarity, n=%d" % n_bits)

    ax2.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        right=False,
        labelbottom=True,
        labelleft=True,
        labelright=False)
    ax2.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax2.set_title("Scalar Bit Similarity, n=%d, w=%d" % (n_bits, w))
    for k in range(len(ref_points)):
        ax2.plot(X_points, scores[:, k], color=colors[k], label=float(ref_points[k]))

    ax2.get_shared_x_axes().join(ax2, ax0)
    ax2.get_shared_x_axes().join(ax2, ax1)

    ax2.set_xlim(0,1)
    ax2.xaxis.set_major_locator(ticker.LinearLocator(5))


    for k in range(len(ref_points)):
        #ax4.plot(X_points, place_scores[:, k], color=colors[k], label=float(ref_points[k]))
        ax2.axvline(x=ref_points[k], ymin=-0.1, ymax=3.4, alpha=0.4, linewidth=1.5, color=colors[k], linestyle='--', clip_on=False)

    handles, labels = ax0.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    #legend = fig.legend(handles, labels, title="Reference Value", fontsize=8, title_fontsize=8, bbox_to_anchor=(1.08, 1),
    #           bbox_transform=ax0.transAxes)
    legend = fig.legend(handles, labels, title="Reference Value", bbox_to_anchor=(0.98, 1), bbox_transform=ax1.transAxes)

    # get the width of your widest label, since every label will need
    # to shift by this amount after we align to the right
    #shift = max([t.get_window_extent().width for t in legend.get_texts()])
    #for t in legend.get_texts():
    #    t.set_ha('left')  # ha is alias for horizontalalignment
        #t.set_position((shift, 0))

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')


if __name__ == "__main__":

    lower_k = 4
    upper_k = 12

    # upper_w = 7
    upper_w = 6
    lower_w = 1

    # number of bins per partition
    partition_sizes = list(range(lower_k, upper_k + 1))
    # partition_sizes = list(range(8,9))
    # partition_sizes = list(range(5, 6))

    # w_sizes = list(range(lower_w, upper_w + 1))

    # partition_sizes = [5,7,9,11]
    # w_sizes = list(range(1,4))
    # partition_sizes = [6, 7, 8, 9, 10, 12]
    # partition_sizes = [16]
    # partition_sizes = [4,8,16,32]
    # partition_sizes = [4, 8]
    # partition_sizes = [2, 4, 8, 16, ]
    partition_sizes = [8, 16, ]
    # partition_sizes = [4,]
    # w_sizes = list(np.arange(0, partition_sizes[0], 8))[1:]
    # w_sizes = list(range(1, 9))
    # w_sizes = list(range(3, 9))
    w_sizes = list(range(3, 4))
    # w_sizes = list(range(1, 2))

    # partition_sizes = [2, 4, 8, 16, 32]

    # primes
    # partition_sizes = [2, 3, 5, 7, 11,13,17]
    # partition_sizes = [5, 7, 11, 13, 17]
    # prime_partition_sizes = [2, 3, 5, 7, 11, 13, 17, 23]
    # for n_grids in range(1, 9):
    # for n_grids in range(1, len(partition_sizes) + 1):
    # for n_grids in range(len(partition_sizes), len(partition_sizes) + 1):

    file_dir = "out/"
    for n_grids in range(2, 3):

        print("partition_sizes:", partition_sizes)

        k_comb = list(it.combinations(partition_sizes, n_grids))
        print("k_comb:")
        print(k_comb)

        # encoder param combos
        # remove where w >= n
        n_w_combos = list(it.product(partition_sizes, w_sizes))
        n_w_combos = [s for s in n_w_combos if s[1] < s[0]]
        print("n_w_combos:")
        for n_w in n_w_combos:
            print(n_w)

        for parts in k_comb:

            for w in w_sizes:

                if 2 * w <= min(parts):
                    encoder_params = []
                    for n in parts:
                        encoder_params.append((n, w))

                    print("parts:", parts)
                    print("encoder_params:", encoder_params)
                    # create filename
                    file_name = "out/%02u_parallel_encoders_n_w" % n_grids
                    for k in parts:
                       file_name += "__%02u_%02u" % (k, w)
                    file_name += ".png"
                    plot_interval_graph_3(encoder_params=encoder_params, file_name=file_name)
                    plt.clf()

                    # plot_similarity_1(n_bits=8, file_dir=file_dir)
                    # plot_similarity_1(n_bits=12, file_dir=file_dir)
                    #plot_encoding_1(n_bits=8, file_dir=file_dir)
                    plot_similarity_2(n_bits=8, file_dir=file_dir)
                    plt.clf()
