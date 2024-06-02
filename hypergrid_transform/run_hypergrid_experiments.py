import numpy as np
import pandas as pd

from collections.abc import Iterable
# matplotlib
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from matplotlib import ticker

# import time

# import xgboost as xgb

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, Normalizer, FunctionTransformer

from brainblocks.tools import HyperGridTransform
from brainblocks.metrics import gnome_similarity




def plot_cosine_similarity(results, title_str, filename_str):
    # create the subplots with the previously created figure "1"
    figure = plt.figure(num=1, figsize=(11, 5), constrained_layout=True)
    f, axes = plt.subplots(1, len(results), num=1)

    for i in range(len(results)):

        result_per_config = results[i]

        X_gnomes, ref_gnomes, scores, X, ref_points, y, le, data_name, config = result_per_config

        # axes for this piece of data
        if isinstance(axes, Iterable):
            ax = axes[i]
        else:
            ax = axes

        ax.plot(X, scores)

        # for pnt in ref_points:
        #    ax.axvline(x=pnt, color='k')

        #         ax.set_xticks(())
        #         ax.set_yticks(())

        if i > 0:
            # ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        ax.set_title("%d grids" % config['n_grids'])

    # figure title
    figure.suptitle(title_str + "\n")

    # save figure plot to file
    plt.savefig(filename_str, bbox_inches='tight')
    plt.close()


def plot_cartesian_gnome_compare(results, title_str, filename_str):
    # create the subplots with the previously created figure "1"
    # figure = plt.figure(num=1, figsize=(11, 5), constrained_layout=True)
    figure = plt.figure(num=1, figsize=(11, 7), constrained_layout=True)
    f, axes = plt.subplots(2, len(results), num=1)

    for i in range(len(results)):

        result_per_config = results[i]

        X_gnomes, ref_gnomes, scores, X, ref_points, y, le, data_name, config = result_per_config

        n_grids = config['n_grids']
        n_bins = config['n_bins']
        n_subspace_dims = config['n_subspace_dims']

        # axes for this piece of data
        ax = axes[0][i]

        ax.plot(X, scores)  # , label=list(ref_points))

        # for pnt in ref_points:
        #    ax.axvline(x=pnt, color='k')

        #         ax.set_xticks(())
        #         ax.set_yticks(())

        if i > 0:
            ax.get_yaxis().set_ticks([])
        # ax.get_xaxis().set_ticks([])
        ax.set_title("%d grids / %d bins" % (n_grids, n_bins))

        # axes for this piece of data
        ax = axes[1][i]

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
        labelsizes = [12, 12, 12, 8]
        ax.tick_params(axis='y', which='minor', labelsize=labelsizes[i])
        ax.tick_params(axis='y', which='major', length=25, direction='out')
        ax.tick_params(axis='y', which='minor', length=0)
        mags = list(config['subspace_periods'].reshape(-1))
        mags = ["%.2f" % mag for mag in mags]
        # mags.reverse()
        ax.yaxis.set_minor_locator(ticker.IndexLocator(n_bins, n_bins // 2))
        ax.yaxis.set_minor_formatter(ticker.FixedFormatter(mags))

        # tick_locations = np.arange(start=-0.5, stop=n_grids * n_bins * n_subspace_dims, step=n_bins)
        # ax.set_yticks(tick_locations)
        # ax.set_yticklabels([])

    # create the legend on the figure, not the axes
    handles, labels = axes[1][-1].get_legend_handles_labels()
    figure.legend(handles, labels, title="Comparison\nPoints", loc='upper right',
                  bbox_to_anchor=(1.1, 0.9))  # , fontsize=20)

    # figure title
    figure.suptitle(title_str + "\n")

    # save figure plot to file
    plt.savefig(filename_str, bbox_inches='tight')
    plt.close()


def run_1D_experiment(datasets):
    results = []

    for data in datasets:

        filename_str = "hypergrid_experiment_%s.png"
        # title_str = "Hypergrid"
        title_str = "Semantic Similarity of Transformed Scalars\n(Grids With Evenly Spaced Periods)"

        X, y, X_plot, ref_points, le, data_name = data

        print("data_name=", data_name)

        # for k in range(1, 5):
        # for k in [1, 2, 4, 8, 16, 32, 64]:
        for k in [1, 2, 4, 16]:
            # ref_points = np.array([[-1.5], [-0.5], [0], [0.5], [1.0]])
            # ref_points = np.array([[-0.5], [0.5]])
            # ref_points = np.array([[-0.5]])
            # ref_points = np.array([[0.0]])

            X_gnomes, ref_gnomes, gridTransform = encode_1D_basis(X, ref_points, n_grids=k, n_subspace_dims=1)

            scores = gnome_similarity(X_gnomes, ref_gnomes)

            results.append((X_gnomes, ref_gnomes, scores, X, ref_points, y, le, data_name,
                            {'n_grids': k, 'n_bins': gridTransform.n_bins,
                             'n_subspace_dims': gridTransform.n_subspace_dims,
                             'subspace_periods': gridTransform.subspace_periods}))

        # Plot the results of the experiments
        print("plot to file", filename_str % data_name)
        # plot_cosine_similarity(results, title_str, filename_str % data_name)
        plot_cartesian_gnome_compare(results, title_str, filename_str % data_name)


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


def standard_basis_encode(X, ref_points):
    n_input_dims = X.shape[1]
    n_grids = 1
    n_subspace_dims = 1

    configs = {"n_input_dims": n_input_dims, "n_grids": n_grids, "n_bins": 8, "n_acts": 1,
               "n_subspace_dims": n_subspace_dims,
               "flatten_output": True, "max_period": 10.0, "min_period": 0.05,
               "use_normal_dist_bases": False,
               "use_standard_bases": True,
               "use_orthogonal_bases": False,
               }

    configs["origin"] = tuple([0.0 for k in range(configs["n_input_dims"])])

    # instantiate Grid Encoder with initial values
    gridTransform = HyperGridTransform(**configs)

    ref_gnomes = gridTransform.transform(ref_points)
    X_gnomes = gridTransform.transform(X)

    return X_gnomes, ref_gnomes, gridTransform


def normal_basis_encode(X, ref_points):
    n_input_dims = X.shape[1]
    n_grids = 100
    n_subspace_dims = 1

    configs = {"n_input_dims": n_input_dims, "n_grids": n_grids, "n_bins": 8, "n_acts": 1,
               "n_subspace_dims": n_subspace_dims,
               "flatten_output": True, "max_period": 10.0, "min_period": 0.05,
               "use_normal_dist_bases": True,
               "use_standard_bases": False,
               "use_orthogonal_bases": False,
               }

    configs["origin"] = tuple([0.0 for k in range(configs["n_input_dims"])])

    # instantiate Grid Encoder with initial values
    gridTransform = HyperGridTransform(**configs)

    ref_gnomes = gridTransform.transform(ref_points)
    X_gnomes = gridTransform.transform(X)

    return X_gnomes, ref_gnomes, gridTransform


def load_1D_data(min_period=0.05):
    # load pandas dataframes of raw data and metadata
    # input vector
    inputs = []
    # outputs = []
    # for index_value in np.arange(-2.5, 2.5, 0.01):
    # for index_value in np.arange(-2.5, 2.5, 0.04):
    # for index_value in np.arange(-5.0, 5.0, 0.04):
    #for index_value in np.arange(-5.0, 5.0, min_period):
    for index_value in np.arange(0, min_period, min_period):
        inputs.append([index_value])
        # inputs.append([index_value,index_value])

        # if abs(index_value) < 0.1:
        #    outputs.append("YES")
        # else:
        #    outputs.append("NO")

    # ref_points = np.array([[-1.5], [-0.5], [0], [0.5], [1.0]])
    ref_points = np.array([[-4.0], [-1.5], [1.5], [4.0]])
    # ref_points = np.array([[-0.5], [0.5]])
    # ref_points = np.array([[-0.5]])
    # ref_points = np.array([[0.0]])
    # ref_points = np.array([[-0.5], [0.5]])

    # print(inputs)
    # print(outputs)
    # exit()

    data_name = "1D"

    print("data loaded %d points" % len(inputs))
    print("using data_name:", data_name)

    # input data
    X = np.array(inputs)

    print("ref_points")
    class_indices = []
    for pnt in ref_points:
        # find index of minimum difference to reference points
        idx = np.abs(np.subtract(X, pnt)).argmin()
        class_indices.append(idx)

    outputs = np.zeros(X.shape)
    for k in range(len(class_indices)):
        outputs[class_indices[k]] = k + 1

    print("ref_points")
    print(class_indices)
    # print(outputs
    print(outputs.flatten().shape)
    # exit()

    # data labels converted to integers
    le = LabelEncoder()
    y = le.fit_transform(outputs.flatten())

    # plottable data points
    X_plot = X

    return X, y, X_plot, ref_points, le, data_name


if __name__ == "__main__":
    # load datasets for experiment
    datasets = [load_1D_data()]

    # run experiments
    run_1D_experiment(datasets)
