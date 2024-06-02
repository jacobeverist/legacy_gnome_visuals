import random
import numpy as np
from scipy.stats import ortho_group
import time

# 1D cyclic scalar encoding parameters
# 1) magnitude
# 2) normalized basis vector
# 2) number of total bins in dimension   (n_bins)
# 3) number of active bits in dimension (n_acts)
# 4) number of bits overlapped between bins (n_bits_overlap = n_acts - 1)
# 5) bits stepped per bin (n_bits_per_bin = 1)
# 6) total bits  (n_bits = n_bins)

# TODO
# 6) create plot of similarity and sampled points, (uniform 1D, 2D, and randomly distributed, vary params)

# 4) make this an sklearn-like interface
# 4a) autodetect input dimensionality based on initial input
# 5) create a cosine similarity sklearn-like class
# 7) create an ensemble similarity and k-WTA class
# 8) create a DPC sklearn estimator in python

# low priority
# 9) add ability to select amount of overlap between bins, or n_bits_per_bin
# 10) be able to have variable parameters for each subspace and each basis, i.e. different n_bins, n_acts, and n_bits_per_bin
# 11) remove loop over number of grid subspaces, and make single numpy operation
# 12) add classic hexagonal grid option


# 13) Fix origin negative drop-off, simultaneous bit change when going to -0.00001 for every grid
#  Perhaps, initialize origin to the center of the first grid bin?

# 1D inputs and 1D grid subspaces with uniformly varying scales from 0.05 to 10.0
#
# displacement, dot product
# -0.6 1
# -0.5 3
# -0.4 0
# -0.3 3
# -0.2 0
# -0.1 0
# 0.0 80
# 0.1 74
# 0.2 76
# 0.3 72
# 0.4 68

np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=200,
                    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})


def otsuka_similarity(X_gnomes, ref_gnomes):
    # def otsuka(x, N=10):
    # - define semantic similarity as Otsuka-Ochiai coefficient
    # 		- v1 . v2 / sqrt( |v1|_1 * |v2|_1 )
    # return x / np.sqrt(N*N)

    #print(X_gnomes.shape)
    #print(ref_gnomes.shape)

    #print("dot product")
    # dot product of bit vectors
    dot_product = np.dot(X_gnomes.astype(int), ref_gnomes.T.astype(int))
    #print(dot_product, dot_product.shape)

    #print("norms")
    l1_norm_1 = np.count_nonzero(X_gnomes, axis=1)
    #print(l1_norm_1, l1_norm_1.shape)

    l1_norm_2 = np.count_nonzero(ref_gnomes, axis=1)
    #print(l1_norm_2, l1_norm_2.shape)

    # sum_of_squares = np.add(square1.reshape(-1, 1), square2.reshape(1, -1))
    #print("product of magnitudes")
    product_of_magnitudes = np.multiply(l1_norm_1.reshape(-1, 1), l1_norm_2.reshape(1, -1))
    #print(product_of_magnitudes, product_of_magnitudes.shape)

    #print("normalizing denominator")
    normalizing_denominator = np.sqrt(product_of_magnitudes)
    #print(normalizing_denominator, normalizing_denominator.shape)

    #print("normalized scores")
    normalized_scores = np.divide(dot_product, normalizing_denominator)
    #print(normalized_scores, normalized_scores.shape)

    return normalized_scores


def tanimoto_similarity(X_gnomes, ref_gnomes):
    # def tanimoto(x, N=10):
    # 	- define semantic similarity as Tanimoto coefficient
    # 		- v1 . v2 / ( |v1|^2 + |v2|^2 - v1 . v2)
    # return x / ( N + N - x)

    #print(X_gnomes.shape)
    #print(ref_gnomes.shape)

    #print("dot product")
    # dot product of bit vectors
    dot_product = np.dot(X_gnomes.astype(int), ref_gnomes.T.astype(int))
    #print(dot_product, dot_product.shape)

    #print("norms")
    l2_norm_1 = np.linalg.norm(X_gnomes.astype(int), axis=1)
    #print(l2_norm_1, l2_norm_1.shape)
    l2_norm_2 = np.linalg.norm(ref_gnomes.astype(int), axis=1)
    #print(l2_norm_2, l2_norm_2.shape)

    #print("squares")
    square1 = np.square(l2_norm_1)
    #print(square1, square1.shape)
    square2 = np.square(l2_norm_2)
    #print(square2, square2.shape)

    #print("sum of squares")
    sum_of_squares = np.add(square1.reshape(-1, 1), square2.reshape(1, -1))
    #print(sum_of_squares, sum_of_squares.shape)
    # sum_of_squares2 = square1 + square2
    # print(sum_of_squares2, sum_of_squares2.shape)

    #print("denominator")
    normalizing_denominator = sum_of_squares - dot_product
    #print(normalizing_denominator, normalizing_denominator.shape)

    #print("scores normalized")
    normalized_scores = np.divide(dot_product, normalizing_denominator)
    #print(normalized_scores, normalized_scores.shape)

    return normalized_scores


def weighted_similarity(X_gnomes, ref_gnomes):
    dot_product = np.dot(X_gnomes.astype(int), ref_gnomes.T.astype(int))

    #print("norms")
    l1_norm_1 = np.count_nonzero(X_gnomes, axis=1)
    #print(l1_norm_1, l1_norm_1.shape)

    l1_norm_2 = np.count_nonzero(ref_gnomes, axis=1)
    #print(l1_norm_2, l1_norm_2.shape)

    #print("sum of norms")
    sum_of_norms = np.add(l1_norm_1.reshape(-1, 1), l1_norm_2.reshape(1, -1))
    #print(sum_of_norms, sum_of_norms.shape)

    #print("half of sum")
    half_sum = sum_of_norms / 2
    #print(half_sum, half_sum.shape)

    #print("scores normalized")
    normalized_scores = np.divide(dot_product, half_sum)
    #print(normalized_scores, normalized_scores.shape)

    return normalized_scores


def gnome_similarity(X_gnomes, ref_gnomes):
    """
    Asymmetric similarity coefficient.

    Values with respect to X_gnomes.

    :param X_gnomes:
    :param ref_gnomes:
    :return:
    """
    sum_scores = np.dot(X_gnomes.astype(int), ref_gnomes.T.astype(int))
    l1_norm = np.count_nonzero(X_gnomes, axis=1)
    normalized_scores = np.divide(sum_scores, l1_norm[:, np.newaxis])

    return normalized_scores


def print_binary(states):
    # print("output shapes)
    # print("received", states.shape)
    # print("reshaped to", states.shape)

    # states = states.reshape(configs["n_grids"], -1)

    for col_index in range(states.shape[0]):
        val_str = "".join(["X" if x else "-" for x in states[col_index, :]])
        print(val_str)

    # states = states.reshape(configs["n_grids"], configs["n_bins"], -1)
    # print("reshaped to", states.shape)

    # for col_index in range(states.shape[0]):
    #    for row_index in range(states.shape[1]):
    # pretty print the values of the encoded binary vector
    #        val_str = "".join(["X" if x else "-" for x in states[col_index, row_index, :]])
    #        print(val_str)
    #    print()


class HyperGridTransform(object):

    def __init__(self, n_input_dims=2, n_bins=4, n_acts=1, n_grids=100, n_subspace_dims=1, origin=None,
                 max_mag=2.0, min_mag=0.05,
                 use_orthogonal_bases=False, use_normal_dist_bases=False, use_standard_bases=False,
                 set_bases=None, set_magnitudes=None, use_random_uniform_magnitudes=False,
                 use_evenly_spaced_magnitudes=False,
                 flatten_output=False, rand_seed=None):

        # check configuration constraints
        if n_subspace_dims > n_input_dims:
            raise Exception(
                "Target subspace dimension must be less than input space dimension.  Received input %d projected into %d" % (
                    n_input_dims, n_subspace_dims))

        # check max and min magnitude constraints
        if max_mag <= min_mag:
            raise Exception("Max magnitude must be greater than min magnitude")

        # random vector basis method
        if [use_normal_dist_bases, use_orthogonal_bases, use_standard_bases, set_bases is not None].count(True) > 1:
            raise Exception("Must choose only one option for subspace vector bases")

        if [use_normal_dist_bases, use_orthogonal_bases, use_standard_bases, set_bases is not None].count(True) == 0:
            use_normal_dist_bases = True

        # origin vector, all vectors are displaced from this point
        if origin is None:
            self.origin = tuple([0.0 for k in range(self.n_input_dims)])
        else:
            if len(origin) != n_input_dims:
                raise Exception("Origin vector must be same dimension as n_input_dims")

            self.origin = origin

        self.origin = np.array(origin)

        # dimensionality
        self.n_input_dims = n_input_dims

        # whether to flatten returned binary matrix or preserve multi-dimensionality
        self.flatten_output = flatten_output

        # number of bins
        self.n_bins = n_bins

        # number of active bits per bin
        self.n_acts = n_acts

        # number of vectors m
        self.n_grids = n_grids
        self.n_subspace_dims = n_subspace_dims

        self.total_n_bits = self.n_grids * self.n_bins

        # verify custom bases shape
        if set_bases is not None:
            # subspace_vectors: (n_grids, n_subspace_dims, n_input_dims)
            custom_shape = set_bases.shape
            if [custom_shape[0] == n_grids, custom_shape[1] == n_subspace_dims,
                custom_shape[2] == n_input_dims].count(False) > 0:
                raise ("custom bases is wrong shape, should be", (n_grids, n_subspace_dims, n_input_dims))

        # verify custom magnitudes shape
        if set_magnitudes is not None:
            # subspace_mags must be (self.n_grids, self.n_subspace_dims)
            custom_shape = set_magnitudes.shape
            if [custom_shape[0] == n_grids, custom_shape[1] == n_subspace_dims].count(False) > 0:
                raise ("custom magnitudes is wrong shape, should be", (n_grids, n_subspace_dims))

        # subspace_mags must be (self.n_grids, self.n_subspace_dims)
        if set_magnitudes is not None:
            # Custom Magnitudes

            ###########
            self.subspace_mags = set_magnitudes

            # set_bases=None, set_magnitudes=None, use_random_uniform_magnitudes=False,
            # use_evenly_spaced_magnitudes=False,
        elif use_random_uniform_magnitudes:
            # Random Magnitudes
            # random magnitudes for subspace basis vectors
            ###########
            self.subspace_mags = np.random.uniform(low=min_mag, high=max_mag, size=self.n_grids * self.n_subspace_dims) \
                .reshape(self.n_grids, self.n_subspace_dims)

        else:
            self.subspace_mags = np.linspace(min_mag, max_mag, endpoint=False, num=n_grids * n_subspace_dims + 1)[1:]
            self.subspace_mags = self.subspace_mags.reshape(n_grids, n_subspace_dims)

        if use_orthogonal_bases:

            # Random Orthonormal Basis Vectors
            # first n_subspace_dims vectors become our basis of each subspace
            ###########

            self.random_orthonormal_vectors = ortho_group.rvs(self.n_input_dims, self.n_grids)[:, :n_subspace_dims, :]

            self.subspace_vectors = self.random_orthonormal_vectors

        elif use_normal_dist_bases:

            # Random Normal Basis Vectors from Normal Distribution
            ###########

            # initialize random seed
            self.rand_state = np.random.RandomState(rand_seed)

            # sample from normal distribution that approximates a n-d sphere
            sphere_vectors = self.rand_state.normal(size=(self.n_grids, self.n_subspace_dims, self.n_input_dims))

            # normalize vectors
            sphere_mags = np.linalg.norm(sphere_vectors, axis=2)
            self.random_normal_vectors = sphere_vectors / sphere_mags[..., np.newaxis]

            self.subspace_vectors = self.random_normal_vectors

        elif use_standard_bases:
            # Standard Basis Vectors
            ###########

            # random standard basis for each subspace
            # random_standard_basis = np.random.permutation(np.identity(self.n_input_dims))[:n_subspace_dims, :]

            """
            self.random_standard_bases = np.array([
               np.random.permutation(np.identity(self.n_input_dims))[:n_subspace_dims, :]
               for k in range(self.n_grids)
            ])
            """

            self.random_standard_bases = np.array([
                np.random.permutation(
                    np.multiply(np.identity(self.n_input_dims), np.random.choice((1, -1), self.n_input_dims))
                )
                [:n_subspace_dims, :]
                for k in range(self.n_grids)
            ])

            # self.subspace_vectors = self.subspace_mags[..., np.newaxis] * self.random_standard_bases
            self.subspace_vectors = self.random_standard_bases

        elif set_bases is not None:
            # subspace_vectors: (n_grids, n_subspace_dims, n_input_dims)
            self.subspace_vectors = set_bases

        else:
            raise Exception("No subspace vector bases selected")

        """
        print("bases")
        print(self.subspace_vectors.shape)
        print("mags")
        print(self.subspace_mags.shape)
        print(self.subspace_mags)
        """

        # print(repr(self.subspace_vectors))
        # print(repr(self.subspace_mags))

    def transform(self, X):

        results = []

        for row in X:
            results.append(self.input(row))

        return np.array(results)

    def input(self, input_vec):

        # input_vec: (n_input_dims,)
        # input_vec: (3,)
        # print("input_vec:", input_vec.shape)

        # subspace_vectors: (n_grids, n_subspace_dims, n_input_dims)
        # subspace_vectors: (4, 2, 3)
        # print("subspace_vectors:", self.subspace_vectors.shape)

        # input vector in coordinate frame defined by origin
        # disp_vec: (n_input_dims, 1)
        # disp_vec: (3, 1)
        disp_vec = input_vec - self.origin
        disp_vec = np.reshape(disp_vec, (-1, 1))

        # print("disp_vec:", disp_vec.shape)

        # matrix multiply @ to get projection to subspaces
        # proj_vec: (n_grids, n_subspace_dims, 1)
        # proj_vec: (4, 2, 1)
        proj_vec = np.matmul(self.subspace_vectors, disp_vec)

        # print("proj_vec:", proj_vec.shape)

        # magnitude of basis vectors of subspace
        # grid_mags = np.linalg.norm(self.subspace_vectors, axis=2)

        # grid_mags: (n_grids, n_subspace_dims, 1)
        # grid_mags: (4, 2, 1)
        grid_mags = self.subspace_mags
        grid_mags = grid_mags[..., np.newaxis]

        # print("grid_mags:", grid_mags.shape)

        # magnitude_displacement_vectors: (n_grids, n_subspace_dims, n_input_dims)
        # magnitude_displacement_vectors: (4, 2, 3)
        # magnitude_displacement_vectors = np.multiply(self.subspace_vectors, grid_mags)
        # print("magnitude_displacement_vectors:", magnitude_displacement_vectors.shape)

        # origin grid center offset
        # mod_vec: (n_grids, n_subspace_dims, 1)
        # mod_vec: (4, 2, 1)
        mod_vec = np.add(proj_vec, grid_mags / (2.0 * self.n_bins))
        # print("mod_vec:", mod_vec.shape)

        # modulo by magnitude to get value within interval
        # np.fmod creates neg remainder, for neg input
        # np.mod forces pos remainder, for neg input
        mod_vec = np.mod(mod_vec, grid_mags)

        # convert to unit interval
        mod_vec = np.divide(mod_vec, grid_mags)

        # indexes for setting bits
        i_bin = np.floor(mod_vec * self.n_bins).astype(np.int).reshape(self.n_grids, self.n_subspace_dims)
        index1 = i_bin

        # subspace grid shape of m-dimensions with n_acts bits per dimension
        subspace_grid_shape = tuple(self.n_bins for k in range(self.n_subspace_dims))

        # initialize all the bits to zero for n_grids subspaces
        bits_matrix = np.zeros((self.n_grids, *subspace_grid_shape), dtype=np.bool)

        # set the bits for each subspace grid
        for grid_index in range(self.n_grids):
            # activation region of subspace grid
            subspace_acts_shape = tuple(self.n_acts for k in range(self.n_subspace_dims))
            subspace_grid_indices = np.indices(subspace_acts_shape)

            # reshape of index offset matrix
            reshape_list = [-1, ] + [1 for k in range(self.n_subspace_dims)]
            reshape_tuple = tuple(reshape_list)

            # offset activation region of subpspace grid
            offset_index = index1[grid_index]
            subspace_grid_indices = subspace_grid_indices + offset_index.reshape(reshape_tuple)

            # modulo the indices to wrap around and prevent ouf range indexes
            subspace_grid_indices = np.mod(subspace_grid_indices, self.n_bins)

            # set the activation region
            bits_matrix[grid_index][tuple(subspace_grid_indices)] = 1

        if self.flatten_output:
            flattened_bits = np.ravel(bits_matrix)
            return flattened_bits
        else:
            return bits_matrix


if __name__ == "__main__":

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

    custom_magnitudes = np.array([
        [1.0, 2.0],
        [2.0, 1.5]
    ])
    """

    # n_input_dims = 1
    # n_grids = 1
    # n_subspace_dims = 1
    custom_bases = np.array([
        [
            [1.]
        ],
        [
            [1.]
        ]
    ])

    print(custom_bases.shape)

    custom_magnitudes = np.array([
        [1.0],
        [1.1]
    ])

    # subspace_vector_bases shape: (n_grids, n_subspace_dims, n_input_dims)
    n_grids = custom_bases.shape[0]
    n_subspace_dims = custom_bases.shape[1]
    n_input_dims = custom_bases.shape[2]

    # configs = {"n_input_dims": 200, "n_grids": 100, "n_bins": 16, "n_acts": 4}
    # configs = {"n_input_dims": 200, "n_grids": 100, "n_bins": 8, "n_acts": 2, "n_subspace_dims": 6,
    #           "flatten_output": True, "max_mag": 2.0, "min_mag": 0.05, "use_normal_dist_bases": False}
    configs = {"n_input_dims": n_input_dims, "n_grids": n_grids, "n_bins": 4, "n_acts": 1,
               "n_subspace_dims": n_subspace_dims,
               "flatten_output": True, "max_mag": 10.0, "min_mag": 0.05,
               "use_normal_dist_bases": False,
               "use_standard_bases": False,
               "use_orthogonal_bases": False,
               "set_bases": custom_bases,
               "set_magnitudes": custom_magnitudes
               }
    # configs["origin"] = tuple([random.uniform(-1.0, 1.0) for k in range(configs["n_input_dims"])])
    configs["origin"] = tuple([0.0 for k in range(configs["n_input_dims"])])

    # instantiate Grid Encoder with initial values
    gridEncoder = HyperGridTransform(**configs)

    # np.random.uniform(low=min_mag, high=max_mag
    origin_vec = np.array([0.0 for i in range(configs["n_input_dims"])])
    origin_state = gridEncoder.input(origin_vec).reshape(configs["n_grids"], -1)

    # input vector
    inputs = []
    for index_value in range(-25, 25):
        input_vec = [0.0 for i in range(configs["n_input_dims"])]
        input_vec[0] = 0.1 * index_value
        input_vec = np.array(input_vec)
        inputs.append(input_vec)

    prev_state = None
    prevInput = None

    print("similarity comparison")
    # print("disp, prev, origin")
    print("disp, origin")
    # input vector
    for input_vec in inputs:

        # print("Execute Input:", input_vec)

        # t0 = time.time()
        states = gridEncoder.input(input_vec)

        # t1 = time.time()
        # print("time: ", t1 - t0)

        # print("output shapes)
        # print("received", states.shape)
        states = states.reshape(configs["n_grids"], -1)
        # print("reshaped to", states.shape)

        if prev_state is not None:
            # print("prev:", prevInput)
            # print_binary(prev_state)
            # print("curr:", input_vec)
            # print_binary(states)
            # print_binary(truth_values)
            truth_values = np.logical_and(prev_state, states)
            prev_similarity_count = np.count_nonzero(truth_values)

            truth_values = np.logical_and(origin_state, states)
            origin_similarity_count = np.count_nonzero(truth_values)

            # print("%.1f" % input_vec[0], prev_similarity_count, origin_similarity_count)
            print("%.1f" % input_vec[0], origin_similarity_count)

        prev_state = states
        prevInput = input_vec

        # print()
