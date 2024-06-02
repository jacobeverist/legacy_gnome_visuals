import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib import colors
import numpy as np
import pandas as pd
import string
import seaborn as sns
from pywaffle import Waffle
import calmap


#######

def test1():
    # plt.clf()

    N = 15
    M = 25
    data = np.random.rand(N, M) * 20

    # create discrete colormap
    cmap = colors.ListedColormap(['gray', 'white'])
    bounds = [0, 10, 20]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, M, 1))
    ax.set_yticks(np.arange(-.5, N, 1))

    plt.savefig("test1.png")  # , bbox_inches='tight')
    plt.close()


#######
def test2():
    # plt.clf()

    N = 15
    M = 25
    # make an empty data set
    # data = np.ones((N, M)) * np.nan

    # fill in some fake data
    # for j in range(3)[::-1]:
    #    data[N // 2 - j: N // 2 + j + 1, N // 2 - j: N // 2 + j + 1] = j

    data = np.zeros(N * M)
    data[::] = np.random.random(N * M)
    data = data.reshape((N, M))

    # make a figure + axes
    # fig, ax = plt.subplots(1, 1, tight_layout=True)
    fig, ax = plt.subplots(1, 1)

    # make color map
    # my_cmap = colors.ListedColormap(['r', 'g', 'b'])

    # set the 'bad' values (nan) to be white and transparent
    # my_cmap.set_bad(color='w', alpha=0)

    # draw the grid
    for x in range(N + 1):
        ax.axhline(x, lw=2, color='k', zorder=5)
    for x in range(M + 1):
        ax.axvline(x, lw=2, color='k', zorder=5)

    # draw the boxes
    # ax.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, M, 0, N], zorder=0)
    # ax.imshow(data, interpolation='none', cmap=my_cmap, zorder=0)
    ax.imshow(data, extent=[0, M, 0, N],
              aspect='equal')  # , interpolation='none', cmap=my_cmap, extent=[0, M, 0, N], zorder=0)

    # turn off the axis labels
    # ax.axis('off')

    plt.savefig("test2.png")  # , bbox_inches='tight')
    plt.close()


#########

def test3():
    # plt.clf()

    # Make a 9x9 grid...
    N = 15
    M = 25
    nrows, ncols = N, M
    image = np.zeros(nrows * ncols)

    # Set every other cell to a random number (this would be your data)
    # image[::2] = np.random.random(nrows*ncols //2 + 1)
    image[::] = np.random.random(nrows * ncols)

    # Reshape things into a 9x9 grid.
    image = image.reshape((nrows, ncols))

    row_labels = range(nrows)
    # col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    col_labels = list(string.ascii_uppercase[:M])
    plt.matshow(image, extent=[0, M, 0, N])
    plt.xticks(range(ncols), col_labels)
    plt.yticks(range(nrows), row_labels)

    # draw the grid
    ax = plt.gca()
    for x in range(N + 1):
        ax.axhline(x, lw=2, color='k', zorder=5)
    for x in range(M + 1):
        ax.axvline(x, lw=2, color='k', zorder=5)

    plt.savefig("test3.png")  # , bbox_inches='tight')
    plt.close()


######


def test4():
    # plt.clf()

    N = 15
    M = 25
    data = pd.DataFrame(np.random.random((N, M)),
                        columns=list(string.ascii_uppercase[:M]))
    # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    # test_list = list(string.ascii_lowercase)

    checkerboard_table(data)

    plt.savefig("test4.png")  # , bbox_inches='tight')
    plt.close()


def checkerboard_table(data, fmt='{:.2f}', bkg_colors=['yellow', 'white']):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    # tb = Table(ax, bbox=[0,0,1,1])
    tb = Table(ax)

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(data):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = bkg_colors[idx]

        tb.add_cell(i, j, width, height, text=fmt.format(val),
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text=label, loc='right',
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height / 2, text=label, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)
    return fig


def test5():
    data = [[66386, 174296, 75131, 577908, 32015],
            [58230, 381139, 78045, 99308, 160454],
            [89135, 80552, 152558, 497981, 603535],
            [78415, 81858, 150656, 193263, 69638],
            [139361, 331509, 343164, 781380, 52269]]

    columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
    rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

    values = np.arange(0, 2500, 500)
    value_increment = 1000

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    cell_colors = plt.cm.binary(np.clip(np.random.rand(len(rows), len(columns)), 0.0, 1.0))

    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          cellColours=cell_colors,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Loss in ${0}'s".format(value_increment))
    plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title('Loss by Disaster')
    plt.savefig("test5.png")  # , bbox_inches='tight')


def test6():
    sns.set(style="white")

    N = 15
    M = 25
    data = np.random.rand(N, M) * 20

    # Generate a large random dataset
    # rs = np.random.RandomState(33)
    # d = pd.DataFrame(data=rs.normal(size=(100, 26)),
    #                 columns=list(string.ascii_letters[26:]))

    # Compute the correlation matrix
    # corr = d.corr()

    # Generate a mask for the upper triangle
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(data, square=True, linewidths=.5)

    # , vmax=.3, center=0,
    #        square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.savefig("test6.png")


def test7():
    # data_list = sns.get_dataset_names()
    # ['anscombe', 'attention', 'brain_networks', 'car_crashes', 'diamonds', 'dots', 'exercise', 'flights', 'fmri',
    # 'gammas', 'iris', 'mpg', 'planets', 'tips', 'titanic']
    # print(data_list)
    # sns.set(style="white")

    # N = 15
    # M = 25
    # data = np.random.rand(N, M) * 20

    # f, ax = plt.subplots(figsize=(11, 9))
    # sns.clustermap(data)
    # plt.savefig("test7.png")

    sns.set()

    iris = sns.load_dataset("iris")
    species = iris.pop("species")
    g = sns.clustermap(iris)
    g.savefig("test7.png")

    return

    # Load the brain networks example dataset
    df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

    # Select a subset of the networks
    used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
    used_columns = (df.columns.get_level_values("network")
                    .astype(int)
                    .isin(used_networks))
    df = df.loc[:, used_columns]

    # Create a categorical palette to identify the networks
    network_pal = sns.husl_palette(8, s=.45)
    network_lut = dict(zip(map(str, used_networks), network_pal))

    # Convert the palette to vectors that will be drawn on the side of the matrix
    networks = df.columns.get_level_values("network")
    network_colors = pd.Series(networks, index=df.columns).map(network_lut)


    # Draw the full plot
    result = sns.clustermap(df.corr(), center=0, cmap="vlag",
                   row_colors=network_colors, col_colors=network_colors,
                   linewidths=.75, figsize=(13, 13))

    result.savefig("test7.png")

def test8():

    data = {'Democratic': 48, 'Republican': 46, 'Libertarian': 3}
    fig = plt.figure(
        FigureClass=Waffle,
        rows=5,
        values=data,
        colors=("#2196f3", "#ff5252", "#999999"),  # shared parameter among subplots
        legend={'loc': 'upper left', 'bbox_to_anchor': (1.1, 1)}
    )

    #fig = plt.figure(
    #    FigureClass=Waffle,
    #    rows=5,
    #    columns=10,
    #    values=[48, 46, 6],
    #    figsize=(5, 3)  # figsize is a parameter of matplotlib.pyplot.figure
    #)
    plt.savefig("test8.png")

    """
    data = pd.DataFrame(
        {
            'labels': ['Hillary Clinton', 'Donald Trump', 'Others'],
            'Virginia': [1981473, 1769443, 233715],
            'Maryland': [1677928, 943169, 160349],
            'West Virginia': [188794, 489371, 36258],
        },
    ).set_index('labels')

    # A glance of the data:
    #                  Maryland  Virginia  West Virginia
    # labels
    # Hillary Clinton   1677928   1981473         188794
    # Donald Trump       943169   1769443         489371
    # Others             160349    233715          36258

    fig = plt.figure(
        FigureClass=Waffle,
        plots={
            '311': {
                'values': data['Virginia'] / 30000,
                'labels': ["{0} ({1})".format(n, v) for n, v in data['Virginia'].items()],
                'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 8},
                'title': {'label': '2016 Virginia Presidential Election Results', 'loc': 'left'}
            },
            '312': {
                'values': data['Maryland'] / 30000,
                'labels': ["{0} ({1})".format(n, v) for n, v in data['Maryland'].items()],
                'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.2, 1), 'fontsize': 8},
                'title': {'label': '2016 Maryland Presidential Election Results', 'loc': 'left'}
            },
            '313': {
                'values': data['West Virginia'] / 30000,
                'labels': ["{0} ({1})".format(n, v) for n, v in data['West Virginia'].items()],
                'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.3, 1), 'fontsize': 8},
                'title': {'label': '2016 West Virginia Presidential Election Results', 'loc': 'left'}
            },
        },
        rows=5,  # shared parameter among subplots
        colors=("#2196f3", "#ff5252", "#999999"),  # shared parameter among subplots
        figsize=(9, 5)  # figsize is a parameter of plt.figure
    )
    """

def test9():

    # Import Data
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/yahoo.csv", parse_dates=['date'])
    df.set_index('date', inplace=True)

    # Plot
    plt.figure(figsize=(16, 10), dpi=80)
    calmap.calendarplot(df['2014']['VIX.Close'], fig_kws={'figsize': (16, 10)},
                        yearlabel_kws={'color': 'black', 'fontsize': 14}, subplot_kws={'title': 'Yahoo Stock Prices'})
    plt.savefig("test9.png")

def test10():
    # sphinx_gallery_thumbnail_number = 2

    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                  "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                        [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                        [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                        [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                        [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                        [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(farmers)))
    ax.set_yticks(np.arange(len(vegetables)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(farmers)
    ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.savefig("test10.png")

if __name__ == '__main__':
    # test1()
    test2()
    test3()
    # test4()
    # test5()
    # plt.clf()
    #test6()
    #test7()
    #test8()
    #test9()
    test10()
