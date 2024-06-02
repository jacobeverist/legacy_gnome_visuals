from intervals import FloatInterval as I
import numpy as np
import itertools as it
import string
from fractions import Fraction

from brainblocks.tools import HyperGridTransform

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

    # place index of each row's on-bit
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
        if overlapped_interval.is_empty():
            return I.empty()

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

        if not intersected_interval.is_empty():
            interval_size = intersected_interval.upper - intersected_interval.lower

            interval_combinations.append((interval_size, intersected_interval, intervals, comb))

    return interval_combinations


def find_self_intersected_combinations(encoder_granulations):
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
            if overlapped_interval.is_empty():
                return I.empty()

        return overlapped_interval
    """

    granulation_keys = list(encoder_granulations.keys())
    granulation_keys.sort(reverse=True)

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

            is_non_empty = not intermediate_interval.is_empty()

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
        # print(comb)

        intervals = []
        bins = []
        for bin, overlap_intervals in comb:

            for interval in overlap_intervals:
                intervals.append(interval)
            bins.append(bin)

        intersected_interval = intersect_intervals(intervals)

        if not intersected_interval.is_empty():
            # print("non-empty", count)
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

        print("%f => %d => %s => %s" % (x, bin_index, format(bin_index, "0%db" % n_bits),
                                        str(np.array([int(k) for k in format(bin_index, "0%db" % n_bits)]))))
        gnomes.append(np.array([int(k) for k in format(bin_index, "0%db" % n_bits)]))

    # reverse bit order so it aligns with base-2 encoding places
    gnome_arrays = np.array(gnomes)
    gnome_arrays = np.flip(gnome_arrays, axis=1)

    print("n_bits=", n_bits)
    print("n_bins=", n_bins)
    print(len(bin_list), "bins")
    return gnome_arrays
    # return np.array(gnomes)


def encode_one_hot(ref_vals, n_bits, x_min=0.0, x_max=1.0):
    interval_to_partition = I.closed(x_min, x_max)

    bin_list = create_n_overlaps_of_w(n_bits, 1, interval_to_partition)

    gnomes = []
    for x in ref_vals:
        if bin_list is None:
            raise Exception("Referenced value %f is not within range [%f,%f]" % (x, x_min, x_max))

        # print("%f => %d => %s" % (x, bin_index, format(bin_index, "0%db" % n_bits)))
        gnomes.append(np.array([1 if x in interval else 0 for interval in bin_list]))

    return np.array(gnomes)


def encode_hypergrid_1d(ref_vals, n_bits=4, w=1):
    custom_bases = np.array([
        np.identity(1)
    ])

    # print(custom_bases)

    custom_periods = np.array([
        [1.0],
    ])

    # self.origin = tuple([0.0 for k in range(self.num_features)])

    # origin =  [w / (2.0 * n_bits)]
    origin = [1 / (2.0 * n_bits)]
    # origin=[0.0]
    # origin =  [-1.0 / (2.0 * n_bits)]
    # origin =  [0.0 / (2.0 * n_bits)]
    hgt = HyperGridTransform(num_grids=1, num_bins=n_bits, num_acts=w, num_subspace_dims=1,
                             set_periods=custom_periods, set_bases=custom_bases, flatten_output=True, origin=origin)
    hgt.fit(ref_vals)
    X_gnomes = hgt.transform(ref_vals)

    return X_gnomes


def encode_with_bins(ref_vals, bin_list):
    gnomes = []
    for x in ref_vals:
        gnomes.append(np.array([1 if x in interval else 0 for interval in bin_list]))

    return np.array(gnomes)


def plot_similarity_2(n_bits=16, w=1, lower_bound=0.0, upper_bound=1.0, file_dir="./"):
    file_name = file_dir + "%02u_%02u_finite_similarity" % (w, n_bits)
    file_name += ".png"

    # unit_interval = I.closed(lower_bound, upper_bound)
    unit_interval = I.closed(0, 1)
    # w = w
    all_bins = create_n_overlaps_of_w(n_bits, w, unit_interval, is_debug=False)

    # reference points for comparison
    ref_points = np.array([[0.21], [0.69]])

    # conversions
    ref_gnomes = encode_with_bins(ref_points, all_bins)
    ref_one_hot = encode_one_hot(ref_points, n_bits)
    # ref_binary_integers = encode_binary_integer(ref_points, n_bits, x_min=lower_bound, x_max=upper_bound)

    X_points = np.arange(lower_bound, upper_bound, 0.001).reshape(-1, 1)

    # conversions
    X_gnomes = encode_with_bins(X_points, all_bins)
    X_one_hot = encode_one_hot(X_points, n_bits)
    # X_binary_integers = encode_binary_integer(X_points, n_bits, x_min=lower_bound, x_max=upper_bound)

    scores = gnome_similarity(X_gnomes, ref_gnomes)
    one_hot_scores = gnome_similarity(X_one_hot, ref_one_hot)
    # binary_scores = gnome_similarity(X_binary_integers, ref_binary_integers)

    # Draw Plots in Each SubAxes

    fig = plt.figure(num=1, figsize=(10, 8), constrained_layout=True)
    fig, axes = plt.subplots(2, 1, num=1)  # , sharex=True)

    ax0 = axes[0]
    ax1 = axes[1]
    # ax2 = axes[2]

    colors = sns.color_palette("Set1", n_colors=len(ref_points))

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
    ax0.set_ylim(0, n_bits)

    draw_encoding(ax0, X_gnomes)

    # for k in range(len(ref_points)):
    #    ax0.plot(X_points, one_hot_scores[:, k], color=colors[k], label=float(ref_points[k]))

    ax0.set_title("Encoding, n=%d" % n_bits)

    ax1.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        right=False,
        labelbottom=True,
        labelleft=True,
        labelright=False)
    ax1.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax1.set_title("Scalar Bit Similarity, n=%d, w=%d" % (n_bits, w))
    for k in range(len(ref_points)):
        ax1.plot(X_points, scores[:, k], color=colors[k], label=float(ref_points[k]))

    # ax1.get_shared_x_axes().join(ax1, ax0)

    ax1.set_xlim(lower_bound, upper_bound)
    ax1.xaxis.set_major_locator(ticker.LinearLocator(5))

    for k in range(len(ref_points)):
        ax1.axvline(x=ref_points[k], ymin=-0.1, ymax=2.1, alpha=0.4, linewidth=1.5, color=colors[k], linestyle='--',
                    clip_on=False)

    handles, labels = ax1.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    legend = fig.legend(handles, labels, title="Reference Value", bbox_to_anchor=(0.98, 1),
                        bbox_transform=ax1.transAxes)

    # get the width of your widest label, since every label will need
    # to shift by this amount after we align to the right

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')


def plot_cyclic_similarity_3(n_bits=16, w=1, lower_bound=0.0, upper_bound=1.0, file_dir="./"):
    file_name = file_dir + "%02u_%02u_cyclic_similarity" % (w, n_bits)
    file_name += ".png"

    # unit_interval = I.closed(lower_bound, upper_bound)
    # all_bins = create_n_overlaps_of_w(n_bits, w, unit_interval, is_debug=False)

    # reference points for comparison
    ref_points = np.array([[0.21], [0.69]])

    # conversions
    ref_gnomes = encode_hypergrid_1d(ref_points, n_bits=n_bits, w=w)
    ref_one_hot = encode_one_hot(ref_points, n_bits)

    X_points = np.arange(lower_bound, upper_bound, 0.001).reshape(-1, 1)

    # conversions
    X_gnomes = encode_hypergrid_1d(X_points, n_bits=n_bits, w=w)
    X_one_hot = encode_one_hot(X_points, n_bits)

    scores = gnome_similarity(X_gnomes, ref_gnomes)
    one_hot_scores = gnome_similarity(X_one_hot, ref_one_hot)
    # binary_scores = gnome_similarity(X_binary_integers, ref_binary_integers)

    # Draw Plots in Each SubAxes

    fig = plt.figure(num=1, figsize=(10, 8), constrained_layout=True)
    fig, axes = plt.subplots(2, 1, num=1)  # , sharex=True)

    ax0 = axes[0]
    ax1 = axes[1]
    # ax2 = axes[2]

    colors = sns.color_palette("Set1", n_colors=len(ref_points))

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
    ax0.set_ylim(0, n_bits)

    draw_encoding(ax0, X_gnomes)
    ax0.set_title("Encoding, n=%d" % n_bits)

    ax1.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=True,
        right=False,
        labelbottom=True,
        labelleft=True,
        labelright=False)
    ax1.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax1.set_title("Scalar Bit Similarity, n=%d, w=%d" % (n_bits, w))
    for k in range(len(ref_points)):
        ax1.plot(X_points, scores[:, k], color=colors[k], label=float(ref_points[k]))

    # ax1.get_shared_x_axes().join(ax1, ax0)

    ax1.set_xlim(lower_bound, upper_bound)
    ax1.xaxis.set_major_locator(ticker.LinearLocator(5))

    for k in range(len(ref_points)):
        ax1.axvline(x=ref_points[k], ymin=-0.1, ymax=2.1, alpha=0.4, linewidth=1.5, color=colors[k], linestyle='--',
                    clip_on=False)

    handles, labels = ax1.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    legend = fig.legend(handles, labels, title="Reference Value", bbox_to_anchor=(0.98, 1),
                        bbox_transform=ax1.transAxes)

    # get the width of your widest label, since every label will need
    # to shift by this amount after we align to the right

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')


if __name__ == "__main__":

    lower_m_grids = 1
    upper_m_grids = 1

    lower_n = 4
    upper_n = 12

    upper_w = 6
    lower_w = 1

    # number of bins per partition
    partition_sizes = list(range(lower_n, upper_n + 1))
    w_sizes = list(range(lower_w, upper_w + 1))

    # primes
    file_dir = "out/"
    for m_grids in range(lower_m_grids, upper_m_grids + 1):

        print(partition_sizes)
        print(m_grids)
        # (num_bins choose num_grids) combinations
        n_comb = list(it.combinations(partition_sizes, m_grids))

        # encoder param combos
        # remove where w >= n
        n_w_combos = list(it.product(partition_sizes, w_sizes))
        n_w_combos = [s for s in n_w_combos if s[1] < s[0]]

        print("n_comb =", n_comb)

        # combination of number of bins per grid
        for parts in n_comb:

            # each value w, same for each grid
            for w in w_sizes:

                # only choose w that is less than half the size of num_bins
                if 2 * w <= min(parts):
                    encoder_params = []
                    for n in parts:
                        encoder_params.append((n, w))

                    plot_similarity_2(n_bits=parts[0], w=w, lower_bound=0.0, upper_bound=1.0, file_dir=file_dir)
                    plt.clf()
                    plot_cyclic_similarity_3(n_bits=parts[0], w=w, lower_bound=0.0, upper_bound=1.0, file_dir=file_dir)
                    plt.clf()
