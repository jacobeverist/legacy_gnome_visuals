# brainblocks
from brainblocks.blocks import ScalarTransformer
from brainblocks.tools import HyperGridTransform
from brainblocks.metrics import *

import numpy as np

# printing boolean arrays neatly
np.set_printoptions(
    precision=3, suppress=True, threshold=1000000, linewidth=400,
    formatter={'bool': lambda bin_val: 'X' if bin_val else '-'})


def run():

    # points for initializing HyperGrid Transform
    points = np.array([[0.0, ]])

    num_grids = 1
    num_bins = 8
    num_subspace_dims = 1
    step_size = 0.04

    # grid for visualizing
    hgt = HyperGridTransform(num_grids=num_grids, num_bins=num_bins, num_subspace_dims=num_subspace_dims,
                             use_standard_bases=True).fit(points)

    # converted feature points to gnomes
    bits_tensor = hgt.transform(points)
    #print(bits_tensor)

    # Scalar Transformer
    n = 64
    w = 8
    max_val = 1.0
    min_val = 0.0
    st = ScalarTransformer(min_val=min_val, max_val=max_val, num_s=n, num_as=w)

    # Parameters about ScalarTransformer

    # there is an extra position beyond max_val due to how ScalarTransformer is built
    # need to fix ScalarTransformer so that stays inside interval [min_val,max_val)
    num_pos = n - w
    period = max_val - min_val
    bin_size = period/num_pos

    # Input Values, placed in middle of bins

    # Range is specified by closed interval and number of points, n-w, equally spaced, middle of bins.
    # xs = np.linspace(min_val+bin_size/2.0, max_val+bin_size/2.0, num=num_pos+1)

    # Range is min plus half bin_size to max plus half bin_size.
    # Add +1 bin_size to max to make sure interval is closed at end.
    xs = np.arange(min_val+bin_size/2., max_val+bin_size, bin_size)

    scalar_gnomes = []
    for p in xs:
        st.set_value(p)
        st.feedforward()
        scalar_gnome = np.array(st.output.bits).astype(bool)
        print(scalar_gnome, "%.4f" % p)
        scalar_gnomes.append(scalar_gnome)

    #ref_gnomes = np.array([scalar_gnomes[0], scalar_gnomes[1]])
    ref_gnomes = np.array(scalar_gnomes)
    scalar_gnomes = np.array(scalar_gnomes)

    int_sim_result = np.dot(scalar_gnomes.astype(int), ref_gnomes.T.astype(int))

    gnome_sim_result = gnome_similarity(scalar_gnomes, ref_gnomes)

    print(int_sim_result)
    #print(gnome_sim_result)


    """
    step_val = 0.001
    xs = np.arange(min_val-step_val, max_val+step_val, step_val)
    gnomes2 = []
    for p in xs:
        st.set_value(p)
        st.feedforward()
        gnome = np.array(st.output.bits).astype(bool)
        if p < min_val + bin_size + step_val:
            print(gnome, "%.4f" % p)
        elif p > max_val - bin_size - step_val:
            print(gnome, "%.4f" % p)

        gnomes2.append(gnome)
    """

    print("n = %d" % n)
    print("w = %d" % w)
    print("cardinality = %d" % num_pos)
    print("interval = [%.4f, %.4f)" % (min_val, max_val))
    print("period = %.4f" % period)
    print("bin_size = %.4f" % bin_size)
    print("points sampled on interval = %d" % len(xs))

    exit()

    x = np.arange(-1.4, 1.4, step_size)
    y = np.arange(-1.4, 1.4, step_size)
    xgrid, ygrid = np.meshgrid(x, y)

    # convert xgrid mesh to flat array
    x_flat = xgrid.reshape(-1, 1)

    # transform list of real values to list of gnome encoding (integer)
    x_transform = hgt.transform(x_flat).astype(int)

    # indices of where bits are set
    indices = np.where(x_transform == 1)

    # give positive integer label for each grid's active bits, where label is the grid number
    x_transform[indices] = indices[2] + 1

    # convert integer labels to same shape as xgrid mesh
    # xgrid_transform = x_transform.reshape(xgrid.shape[0], xgrid.shape[1], num_bins).astype(int)

    #for point_index in range(len(xs)):
    #    point = [(xs[point_index],)]
    for point in xs:

        # assume one point, remove excess axis for single point, convert to int
        X_gnome = hgt.transform(point)
        X_gnome = X_gnome.reshape(X_gnome.shape[1:]).astype(int)

        # give positive integer label for each grid's active bits, where label is the grid number
        #indices = np.where(X_gnomes == 1)
        #X_gnomes[indices] = indices[0] + 1


if __name__ == "__main__":
    run()
