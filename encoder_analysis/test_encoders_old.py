# plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", color_codes=True)

try:
    from encoders.encoders import *
    from encoders.visuals import *
    from encoders.helpers import *
except:
    from encoder_analysis.encoders.encoders import *
    from encoder_analysis.encoders.visuals import *
    from encoder_analysis.encoders.helpers import *


def plot_interval_multi_encoder(encoder, desc_str="Encoder", lower_bound=0.0, upper_bound=1.0, file_dir="./out"):
    w = encoder.w
    n_bits = encoder.n
    l = encoder.l

    markersize = 4

    file_name = file_dir + "%02u_%02u_" % (w, n_bits) + encoder.__class__.__name__ + ".png"

    # reference points for comparison
    ref_points = np.array([[0.21], [0.69]])
    ref_gnomes = encoder.encode(ref_points)

    # sampled points over the space
    # X_points = np.arange(lower_bound, upper_bound, 0.001).reshape(-1, 1)
    X_points = np.array(encoder.region_centers).reshape(-1, 1)
    # X_gnomes = encoder.region_codes
    X_gnomes = encoder.encode(X_points)

    # print(ref_gnomes)
    # print(ref_gnomes.shape)

    # print(X_gnomes)
    # print(X_gnomes.shape)

    # print(X_gnomes2)
    # print(X_gnomes2.shape)

    scores = gnome_similarity(X_gnomes, ref_gnomes)
    scores2 = count_similarity(X_gnomes, ref_gnomes)

    print("scores", scores.shape, scores2.shape)
    print(scores)
    print(scores2)

    boundaries = encoder.region_boundaries
    deltas = encoder.region_deltas

    # Draw Plots in Each SubAxes
    # subplot_kw, gridspec_kw
    fig = plt.figure(num=1, figsize=(10, 8), dpi=300, constrained_layout=True)
    #fig = plt.figure(num=1, figsize=(10, 8), dpi=200, constrained_layout=True)
    # fig, axes = plt.subplots(6, 1, num=1, gridspec_kw={'height_ratios': [1, 0.5, 0.5, 0.5, 0.5, 1]})  # , sharex=True)
    #fig, axes = plt.subplots(4, 1, num=1, gridspec_kw={'height_ratios': [1, 0.7, 0.3, 1]})  # , sharex=True)
    fig, axes = plt.subplots(4, 1, num=1, gridspec_kw={'height_ratios': [1, 1, 1, 1]})  # , sharex=True)

    ax0 = axes[0]
    # ax1 = axes[1]
    ax2 = axes[1]
    # ax2 = axes[2]
    # ax3 = axes[3]
    # ax4 = axes[4]
    # ax5 = axes[5]
    ax4 = axes[2]
    ax5 = axes[3]
    colors = sns.color_palette("Set1", n_colors=len(ref_points))

    # Encoding Plot
    # ax0.set_title("%s, n=%d, w=%d" % (desc_str,n_bits,w))
    # ax0.tick_params(
    #    axis='both',
    #    which='both',
    #    bottom=False,
    #    left=False,
    #    right=True,
    #    labelbottom=False,
    #    labelleft=False,
    #    labelright = True, labelsize = 'small')
    # ax0.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    # ax0.set_ylim(0, n_bits)
    # ax0.set_ylabel("Encoded Bits")
    # draw_encoding(ax0, X_gnomes)

    ax0.set_title("%s, n=%d, w=%d" % (desc_str, n_bits, w))
    ax0.tick_params(
        axis='both',
        which='both',
        #labelbottom=True,
        #bottom=True,
        labelbottom=False,
        bottom=False,
        left=False,
        right=True,
        labelleft=False,
        labelright=True, labelsize='small')
    #ax0.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax0.yaxis.set_major_locator(ticker.IndexLocator(5, 0))
    ax0.set_ylim(-0.5, n_bits + 0.5)
    ax0.set_ylabel("Encoding Bins\non Interval")

    #ax0.set_xlim(lower_bound, upper_bound)
    #ax0.xaxis.set_major_locator(ticker.LinearLocator(5))

    #draw_encoder_bins(ax0, encoder, fontsize=6)
    draw_multi_encoder_bins(ax0, encoder, fontsize=6)

    """
    # Granulation Plot
    # ax1.set_xlim(lower_bound, upper_bound)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_axis_on()
    ax1.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)
    ax1.set_ylabel("Decomposition")
    ax1.get_shared_x_axes().join(ax1, ax0)
    ax1.vlines([0.0, 1.0], ymin=-100, ymax=100, color='k', alpha=0.2)

    draw_decomposition(ax1, boundaries)
    """
    # Features SubPlot (Boundaries, Weight, Crossings)
    ax2.tick_params(
        axis='both',
        which='both',
        #labelbottom=True,
        #bottom=True,
        labelbottom=False,
        bottom=False,
        left=False,
        right=True,
        labelleft=False,
        labelright=True)

    # Data for Features
    bin_weights = encoder.region_weights
    max_bin_weight = max(bin_weights)
    bin_weights_y = np.append(bin_weights, [bin_weights[-1], ])

    # scale y-axis properties
    ax2.set_ylim(-1, max_bin_weight + 2)
    ax2.yaxis.set_major_locator(ticker.IndexLocator(5, 0))
    ax2.set_ylabel("Features of\nEncodings")

    # share ax0 and ax2 x-axis
    ax2.get_shared_x_axes().join(ax2, ax0)

    # set central ordinal value on y-axis for swarmplot
    bottom, top = ax2.get_ylim()
    swarm_ordinal = (top-bottom)/2.0 + bottom

    # points repeated for each delta count
    repeat_boundaries = []
    for k in range(len(deltas)):
        count = deltas[k]
        for j in range(count):
            repeat_boundaries.append(boundaries[k])

    # ordinal number that we want swarm points to be plotted on the y-axis
    y_vals = [swarm_ordinal for k in range(len(repeat_boundaries))]

    # FIXME: hacked seaborn by adding a second category with ordinal number out of plot range.
    #  Enables swarm points beyond unit category range
    #  This point is plotted, but not seen since '-100' is far out of axes y-range
    repeat_boundaries.append(0.0)
    y_vals.append(-100)

    # do swarm plot
    sns.swarmplot(x=repeat_boundaries, y=y_vals, orient='h', color=colors[0], ax=ax2, size=6, native_scale=True,
                  legend=False, label="Crossings")

    # remove extra category
    repeat_boundaries.pop(-1)
    y_vals.pop(-1)

    # draw grid lines representing boundaries between regions
    ax2.axvline(x=boundaries[0], ymin=bottom, ymax=top, alpha=0.2, linewidth=0.5, color='k', zorder=-1,
                label="Boundary")
    for k in range(1, len(boundaries)):
        ax2.axvline(x=boundaries[k], ymin=bottom, ymax=top, alpha=0.2, linewidth=0.5, color='k', zorder=-1)

    # draw gnome weights
    ax2.step(boundaries, bin_weights_y, where='post', color=colors[1], alpha=0.6, zorder=1, label="Weight")
    ax2.fill_between(boundaries, -1, bin_weights_y, step='post', color=colors[1], alpha=0.3, zorder=1)


    handles1, labels1 = ax2.get_legend_handles_labels()

    # remove the hackish category label, so we only have one "Delta" in the legend
    min_category = -1
    min_elements = 1e100
    for k in range(len(labels1)):
        if labels1[k] == "Crossings":
            handle = handles1[k]
            num_points = len(handle.get_offsets())
            if num_points < min_elements:
                min_elements = num_points
                min_category = k
    handles1.pop(min_category)
    labels1.pop(min_category)

    # plot legend for property data
    #legend = fig.legend(handles1, labels1, bbox_to_anchor=(0.98, 1), bbox_transform=ax2.transAxes)
    legend = ax2.legend(handles1, labels1, title="Features", ncol=3, fontsize=8, title_fontsize=9)

    """
    from matplotlib.bezier import BezierSegment
    
    # draw delta count
    # ax2.bar(boundaries, deltas, width=0.005, linewidth=0, color='k')
    ax2.plot(boundaries, deltas, color=colors[0], alpha=0.6, zorder=2, label="Delta")
    ax2.fill_between(boundaries, 0, deltas, color=colors[0], alpha=0.3, zorder=2)

    # markers
    ax2.plot(boundaries, deltas, marker='o', markersize=markersize, color=colors[0], linestyle='', alpha=0.6, zorder=2)
    """

    """
    # draw delta count with bezier curve between spikes
    for k in range(0, len(boundaries) - 1):
        max_delta = max(deltas[k], deltas[k+1])
        control_points = [(boundaries[k], deltas[k]),
                          (boundaries[k], 0),
                          (boundaries[k + 1], 0),
                          #(boundaries[k], deltas[k]-max_delta),
                          #(boundaries[k + 1], deltas[k+1]-max_delta),
                          (boundaries[k + 1], deltas[k + 1])]

        bs = BezierSegment(control_points)
        tnew = np.linspace(0, 1, num=25)
        retvals = bs(tnew)
        xnew = retvals[:, 0]
        ynew = retvals[:, 1]
        if k == len(boundaries)-2:
            ax2.plot(xnew, ynew, color=colors[0], alpha=0.6, zorder=2, label="Delta")
        else:
            ax2.plot(xnew, ynew, color=colors[0], alpha=0.6, zorder=2)
        ax2.fill_between(xnew, 0, ynew, color=colors[0], alpha=0.3, zorder=2)
    """

    # from scipy.interpolate import CubicSpline
    # spl = CubicSpline(boundaries, deltas)
    # xnew = np.linspace(0, 1, num=51)
    # ax2.plot(xnew, spl(xnew, nu=0), color=colors[0], alpha=0.6, zorder=2, label="Delta")

    # import matplotlib.path as mpath
    # import matplotlib.patches as mpatches
    # Path = mpath.Path
    # for k in range(0, len(boundaries)-1):
    #    pp1 = mpatches.PathPatch(
    #        Path([(boundaries[k], deltas[k]),
    #              (boundaries[k], 0),
    #              (boundaries[k+1], 0),
    #              (boundaries[k+1], deltas[k+1])],
    #             [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
    #        fc="none", color=colors[0], transform=ax2.transData, alpha=0.6, zorder=2, fill="bottom")
    #    ax2.add_patch(pp1)
    # ax2.plot(boundaries, deltas, marker='o', color=colors[0], linestyle='', alpha=0.6, zorder=2, label="Delta")

    # ax2.fill_between(boundaries, 0, deltas)

    # ax2.step(boundaries, bin_weights_y, marker='o', where='post', color=colors[1], alpha=0.6, zorder=1, label="Weight")
    # ax2.plot(boundaries, bin_weights_y, marker='o', markersize=markersize, color=colors[1], linestyle='', alpha=0.6,
    #         zorder=3)

    """
    ax3.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=False,
        right=True,
        labelbottom=False,
        labelleft=False,
        labelright=True)

    ax3.set_ylim(bottom=0, top=max_bin_weight + 0.5)
    ax3.set_ylabel("Weight")
    ax3.yaxis.set_major_locator(ticker.IndexLocator(1, 0))

    # share ax1 and ax3 x-axis
    ax3.get_shared_x_axes().join(ax3, ax2)

    # fix x-axis to interval bounds
    ax3.set_xlim(lower_bound, upper_bound)

    # draw_delta_count(ax3, boundaries, deltas)

    bin_weights_y = np.append(bin_weights, [bin_weights[-1], ])

    # plt.step(x_data, y_data, where='pre', label='vert_first')

    # ax3.step(boundaries, bin_weights_y, where='pre', color='k')
    ax3.step(boundaries, bin_weights_y, where='post', color='k')
    ax3.bar(boundaries, deltas, width=0.005, linewidth=0, color='k')
    # ax3.plot(boundaries, bin_weights_y, color='k')
    # ax3.scatter(boundaries, deltas, color='k')


    for k in range(len(boundaries)):
        ax3.axvline(x=boundaries[k], ymin=-1, ymax=max_bin_weight + 1, alpha=0.2, linewidth=0.5, color='k', zorder=-1)
    """

    # Similarity Plot

    # data to plot
    max_score = np.max(scores2)

    ax4.tick_params(
        axis='both',
        which='both',
        #labelbottom=True,
        #bottom=True,
        labelbottom=False,
        bottom=False,
        left=False,
        right=True,
        labelleft=False,
        labelright=True)

    # scale y-axis to maximum of weight
    ax4.set_ylim(-1, max_score + 2)
    ax4.yaxis.set_major_locator(ticker.IndexLocator(5, 1))
    ax4.set_ylabel("Similarity of\nExample Values")

    # share ax0 and ax4 x-axis
    ax4.get_shared_x_axes().join(ax4, ax0)

    # plot similarity scores for each reference value
    for k in range(len(ref_points)):
        scores_y = np.append(scores2[:, k], [scores2[-1, k], ])
        ax4.step(boundaries, scores_y, where='post', color=colors[k], label=float(ref_points[k]))
        ax4.fill_between(boundaries, -1, scores_y, step='post', color=colors[k], alpha=0.3, zorder=1)

    # draw vertical line indicating reference value on x-axis
    for k in range(len(ref_points)):
        ax4.axvline(x=ref_points[k], ymin=-0.1, ymax=2.1, alpha=1.0, linewidth=1.5, color=colors[k], linestyle='--',
                    clip_on=True)

    # draw boundaries between each region
    for k in range(len(boundaries)):
        ax4.axvline(x=boundaries[k], ymin=-0.1, ymax=2.1, alpha=0.2, linewidth=0.5, color='k', zorder=-1)

    # show legend for each reference value
    handles, labels = ax4.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    #legend = fig.legend(handles, labels, title="Value Similarity", bbox_to_anchor=(0.98, 1), bbox_transform=ax4.transAxes)
    #legend = ax4.legend(handles, labels, title="Value Similarity", bbox_to_anchor=(0.98, 1), bbox_transform=ax4.transAxes)
    legend = ax4.legend(handles, labels, title="Similarity of", ncol=2, fontsize=8, title_fontsize=8)

    """
    repeat_boundaries = []
    for k in range(len(deltas)):
        count = deltas[k]
        for j in range(count):
            repeat_boundaries.append(boundaries[k])
    sns.swarmplot(x=repeat_boundaries, color=colors[0], ax=ax4)

    ax4.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=False,
        right=True,
        labelbottom=True,
        labelleft=False,
        labelright=True)
    ax4.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    """

    # ax2.plot(boundaries, deltas, marker='o', markersize=markersize, color=colors[0], linestyle='', alpha=0.6, zorder=2)
    # tips = sns.load_dataset("tips")
    # sns.swarmplot(data=tips, x="total_bill", y="day", hue="day", legend=False)

    # legend = fig.legend(handles, labels, title="Reference Value", bbox_to_anchor=(0.98, 1),
    #                    bbox_transform=ax4.transAxes)

    # Granulation Plot
    # ax1.set_xlim(lower_bound, upper_bound)

    ax5.tick_params(
        axis='both',
        which='both',
        labelbottom=True,
        bottom=True,
        #labelbottom=False,
        #bottom=False,
        left=False,
        right=True,
        labelleft=False,
        labelright=True, labelsize='small')
    #ax5.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax5.yaxis.set_major_locator(ticker.IndexLocator(5, 0))
    ax5.set_ylim(-0.5, n_bits + 0.5)
    ax5.set_ylabel("Bit Encoding vs.\nReal Value")
    ax5.get_shared_x_axes().join(ax5, ax0)

    draw_bits_by_data(ax5, encoder)
    # draw_encoder_bins(ax5, encoder)

    ax5.set_xlim(lower_bound, upper_bound)

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')


def plot_interval_encoder(encoder, desc_str="Encoder", lower_bound=0.0, upper_bound=1.0, file_dir="./out"):
    w = encoder.w
    n_bits = encoder.n

    file_name = file_dir + "%02u_%02u_" % (w, n_bits) + encoder.__class__.__name__ + ".png"

    # reference points for comparison
    ref_points = np.array([[0.21], [0.69]])
    ref_gnomes = encoder.encode(ref_points)

    # sampled points over the space
    # X_points = np.arange(lower_bound, upper_bound, 0.001).reshape(-1, 1)
    X_points = np.array(encoder.region_centers).reshape(-1, 1)
    # X_gnomes = encoder.region_codes
    X_gnomes = encoder.encode(X_points)

    print(ref_gnomes)
    print(ref_gnomes.shape)

    print(X_gnomes)
    print(X_gnomes.shape)

    # print(X_gnomes2)
    # print(X_gnomes2.shape)

    scores = gnome_similarity(X_gnomes, ref_gnomes)

    boundaries = encoder.region_boundaries
    deltas = encoder.region_deltas

    # Draw Plots in Each SubAxes
    # subplot_kw, gridspec_kw
    fig = plt.figure(num=1, figsize=(10, 8), dpi=200, constrained_layout=True)
    fig, axes = plt.subplots(6, 1, num=1, gridspec_kw={'height_ratios': [1, 0.5, 0.5, 0.5, 0.5, 1]})  # , sharex=True)
    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    ax3 = axes[3]
    ax4 = axes[4]
    ax5 = axes[5]
    colors = sns.color_palette("Set1", n_colors=len(ref_points))

    # Encoding Plot
    # ax0.set_title("%s, n=%d, w=%d" % (desc_str,n_bits,w))
    # ax0.tick_params(
    #    axis='both',
    #    which='both',
    #    bottom=False,
    #    left=False,
    #    right=True,
    #    labelbottom=False,
    #    labelleft=False,
    #    labelright = True, labelsize = 'small')
    # ax0.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    # ax0.set_ylim(0, n_bits)
    # ax0.set_ylabel("Encoded Bits")
    # draw_encoding(ax0, X_gnomes)

    ax0.set_title("%s, n=%d, w=%d" % (desc_str, n_bits, w))
    ax0.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        right=True,
        labelbottom=False,
        labelleft=False,
        labelright=True, labelsize='small')
    ax0.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax0.set_ylim(-0.5, n_bits + 0.5)
    ax0.set_ylabel("Bins")

    draw_encoder_bins(ax0, encoder)

    # Granulation Plot
    # ax1.set_xlim(lower_bound, upper_bound)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_axis_on()
    ax1.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)
    ax1.set_ylabel("Decomposition")
    ax1.get_shared_x_axes().join(ax1, ax0)
    ax1.vlines([0.0, 1.0], ymin=-100, ymax=100, color='k', alpha=0.2)

    draw_decomposition(ax1, boundaries)

    max_change_count = max(encoder.region_weights)

    ax2.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=False,
        right=True,
        labelbottom=False,
        labelleft=False,
        labelright=True)

    ax2.set_ylim(bottom=0, top=max_change_count + 0.5)
    ax2.set_ylabel("Delta Count")
    ax2.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    # ax3.set_ylim(bottom=0, top=max_change_count)

    # share ax1 and ax2 x-axis
    ax2.get_shared_x_axes().join(ax2, ax1)

    # fix x-axis to interval bounds
    ax2.set_xlim(lower_bound, upper_bound)

    # draw_delta_count(ax2, boundaries, deltas)
    # ax2.plot(boundaries, deltas, color='k')
    # ax2.scatter(boundaries, deltas, color='k')
    ax2.bar(boundaries, deltas, width=0.005, linewidth=0, color='k')

    for k in range(len(boundaries)):
        ax2.axvline(x=boundaries[k], ymin=-1, ymax=max_change_count + 1, alpha=0.2, linewidth=0.5, color='k', zorder=-1)

    bin_weights = encoder.region_weights
    max_bin_weight = max(bin_weights)

    ax3.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=False,
        right=True,
        labelbottom=False,
        labelleft=False,
        labelright=True)

    ax3.set_ylim(bottom=0, top=max_bin_weight + 0.5)
    ax3.set_ylabel("Weight")
    ax3.yaxis.set_major_locator(ticker.IndexLocator(1, 0))

    # share ax1 and ax3 x-axis
    ax3.get_shared_x_axes().join(ax3, ax2)

    # fix x-axis to interval bounds
    ax3.set_xlim(lower_bound, upper_bound)

    # draw_delta_count(ax3, boundaries, deltas)

    bin_weights_y = np.append(bin_weights, [bin_weights[-1], ])

    # plt.step(x_data, y_data, where='pre', label='vert_first')

    # ax3.step(boundaries, bin_weights_y, where='pre', color='k')
    ax3.step(boundaries, bin_weights_y, where='post', color='k')
    # ax3.plot(boundaries, bin_weights_y, color='k')
    # ax3.scatter(boundaries, deltas, color='k')

    for k in range(len(boundaries)):
        ax3.axvline(x=boundaries[k], ymin=-1, ymax=max_bin_weight + 1, alpha=0.2, linewidth=0.5, color='k', zorder=-1)

    # Similarity Plot
    ax4.tick_params(
        axis='both',
        which='both',
        bottom=True,
        left=False,
        right=True,
        labelbottom=True,
        labelleft=False,
        labelright=True)
    ax4.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax4.set_ylabel("Similarity")
    for k in range(len(ref_points)):
        # ax4.plot(X_points, scores[:, k], color=colors[k], label=float(ref_points[k]))
        # ax4.step(X_points, scores[:, k], where='mid', marker='o', markersize=4, color=colors[k], label=float(ref_points[k]))
        ax4.step(X_points, scores[:, k], where='mid', color=colors[k], label=float(ref_points[k]))

    # share ax1 and ax2 x-axis
    ax4.get_shared_x_axes().join(ax4, ax3)

    # fix x-axis to interval bounds
    ax4.set_xlim(lower_bound, upper_bound)
    ax4.xaxis.set_major_locator(ticker.LinearLocator(5))

    for k in range(len(ref_points)):
        ax4.axvline(x=ref_points[k], ymin=-0.1, ymax=2.1, alpha=0.4, linewidth=1.5, color=colors[k], linestyle='--',
                    clip_on=True)
        # clip_on = False)

    for k in range(len(boundaries)):
        ax4.axvline(x=boundaries[k], ymin=-0.1, ymax=2.1, alpha=0.2, linewidth=0.5, color='k', zorder=-1)

    handles, labels = ax4.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    # legend = fig.legend(handles, labels, title="Reference Value", bbox_to_anchor=(0.98, 1),
    #                    bbox_transform=ax4.transAxes)

    # Granulation Plot
    # ax1.set_xlim(lower_bound, upper_bound)

    ax5.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        right=True,
        labelbottom=False,
        labelleft=False,
        labelright=True, labelsize='small')
    ax5.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax5.set_ylim(-0.5, n_bits + 0.5)
    ax5.set_ylabel("Bits")
    ax5.get_shared_x_axes().join(ax5, ax4)

    draw_bits_by_data(ax5, encoder)
    # draw_encoder_bins(ax5, encoder)

    if not file_name is None:
        plt.savefig(file_name, bbox_inches='tight')


if __name__ == "__main__":
    file_dir = "../encoder_analysis/out/"

    taper_encoder = TaperingWeightEncoder(n=16, w=3)
    fixed_encoder = FixedWeightEncoder(n=16, w=3)

    multi_encoder = MultiEncoder()
    # multi_encoder.add_encoder(taper_encoder)
    # multi_encoder.add_encoder(fixed_encoder)

    # multi_encoder.add_encoder(FixedWeightEncoder(n=5, w=1))
    # multi_encoder.add_encoder(FixedWeightEncoder(n=7, w=1))
    # multi_encoder.add_encoder(FixedWeightEncoder(n=11, w=1))

    # multi_encoder.add_encoder(FixedWeightEncoder(n=4, w=1))
    # multi_encoder.add_encoder(FixedWeightEncoder(n=16, w=1))
    # multi_encoder.add_encoder(FixedWeightEncoder(n=7, w=3, lower_bound=0.5, L=0.5))
    # multi_encoder.add_encoder(FixedWeightEncoder(n=9, w=3, lower_bound=0.3, L=0.7))
    # multi_encoder.add_encoder(FixedWeightEncoder(n=11, w=3))

    multi_encoder.add_encoder(FixedWeightEncoder(n=7, w=3, L=0.5))
    multi_encoder.add_encoder(FixedWeightEncoder(n=7, w=3, L=0.5))
    multi_encoder.add_encoder(FixedWeightEncoder(n=7, w=3, L=0.5))
    print()
    # multi_encoder.add_encoder(FixedWeightEncoder(n=7, w=3, lower_bound=0.00001, L=0.5))
    multi_encoder.add_encoder(FixedWeightEncoder(n=9, w=3, L=0.7))
    print()
    multi_encoder.add_encoder(FixedWeightEncoder(n=11, w=3))
    print()

    multi_encoder.config()

    # X = np.array([[0.21], [0.69], [0.91]])
    # composite_codes = multi_encoder.encode(X)
    # print(composite_codes)
    # print(composite_codes.shape)

    # print(multi_encoder.upper_bound)
    # print(multi_encoder.lower_bound)
    # print(multi_encoder.L)

    plot_interval_multi_encoder(multi_encoder, "Multi Encoder", file_dir=file_dir)
    plt.clf()

    # plot_interval_encoder(taper_encoder, "Tapering Weight Encoder", file_dir=file_dir)
    # plt.clf()
    # plot_interval_encoder(fixed_encoder, "Fixed Weight Encoder", file_dir=file_dir)
    # plt.clf()
