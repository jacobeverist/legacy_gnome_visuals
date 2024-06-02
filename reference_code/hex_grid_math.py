# Hexagonal Grid Experimentation with coordinate systems
# switching from cartesian space to hex space, and vice versa

import numpy as np

np.set_printoptions(precision=3)


# Johnson-Lindenstrauss random projection

# n = dimensions of original data
# d = dimensions of projected data
# M = d x n matrix of d unit vectors of dimension n
# Vd = M * Vn

# how to ensure all unit vectors are orthogonal?

# selecting elements randomly
# a_ij = {1/sqrt(d), -1/sqrt(d)} with probabilities {1/2, 1/2}
# a_ij = {1/sqrt(d/3), 0.0, -1/sqrt(d/3)} with probabilities {1/6, 2/3, 1/6}

# modifiable or updateable parameters of grid encoder
# origin point: o=[0,0,...,0], initial displacement for location-invariance
# scale factor:  s=[1,1,...,1],  change scale for scale-invariance
# transform matrix: N*N orthogonal matrix R=I,
#  that is a rotation/reflection of object for orientation-invariance
#  (can also be scale/shear if non-orthogonal)


# projection of some vector u onto vector v
def project_vector_to_vector(u, v):
    # magnitude of v
    v_mag = np.linalg.norm(v, axis=-1)

    # dot product of u and v
    v_scalar = np.dot(u, v) / (v_mag * v_mag)

    u_project_to_v = v_scalar * v

    return u_project_to_v


def project_vector_to_plane(u, norm_vec):
    u_project_to_n = project_vector_to_vector(u, norm_vec)

    u_project_to_plane = u - u_project_to_n

    return u_project_to_plane


def cube_to_axial(x, y, z):
    q = x
    r = z
    return q, r


def axial_to_cube(q, r):
    x = q
    z = r
    y = -x - z

    return x, y, z


def cube_round(x, y, z):
    rx = np.round(x)
    ry = np.round(y)
    rz = np.round(z)

    x_diff = np.fabs(rx - x)
    y_diff = np.fabs(ry - y)
    z_diff = np.fabs(rz - z)

    if x_diff > y_diff and x_diff > z_diff:
        rx = -rx - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return rx, ry, rz


def hex_round(q, r):
    x, y, z = axial_to_cube(q, r)
    rx, ry, rz = cube_round(x, y, z)
    rq, rr = cube_to_axial(rx, ry, rz)

    return rq, rr


def hex_to_cartesian(q, r):
    # grid basis vectors

    # (0,0) -> (1,0) in hex axial coordinates
    q_basis = np.array([np.sqrt(3.0), 0.0])
    r_basis = np.array([np.sqrt(3.0) / 2.0, 3.0 / 2.0])
    size = 1.0

    point_cartesian = (q * q_basis + r * r_basis) * size

    return point_cartesian


def cartesian_to_hex(x, y):
    x_basis = np.array([np.sqrt(3.0) / 3.0, 0.0])
    y_basis = np.array([-1.0 / 3.0, 2.0 / 3.0])
    size = 1.0

    point_hex = (x * x_basis + y * y_basis) / size

    return point_hex


if __name__ == "__main__":

    # test_point = [0.0,0.0]

    # center point in grid
    test_point = [np.sqrt(3) + np.sqrt(3) / 2, 3 / 2]
    test_point = [5.4, -3.5]

    # u-vector unit direction
    # np.sqrt(3) * [1,0]

    # v-vector unit direction
    # np.sqrt(3) * [np.cos(np.pi/3), np.sin(np/3)]

    for x_val in np.arange(0.0, 20.0, np.sqrt(3)):
        test_point = [x_val * np.cos(1 * np.pi / 3.), x_val * np.sin(1 * np.pi / 3.)]
        hex_point = cartesian_to_hex(test_point[0], test_point[1])
        print("cart_to_hex", x_val, test_point, hex_point)

        test_point = [x_val, 0.0]
        hex_point = cartesian_to_hex(test_point[0], test_point[1])
        print("cart_to_hex", x_val, test_point, hex_point)

    hex_point = cartesian_to_hex(test_point[0], test_point[1])
    print("hex point:", hex_point)
    print("hex round:", hex_round(hex_point[0], hex_point[1]))

    cartesian_point = hex_to_cartesian(hex_point[0], hex_point[1])
    print("cartesian point:", cartesian_point)

    cartesian_point = hex_to_cartesian(1, 1)
    print("cartesian point:", cartesian_point)

    test_vec = np.array([1.0, 0.0, 0.0])
    norm_vec = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, np.sqrt(2) / 2])
    print("test_vec:", test_vec)
    print("norm_vec:", norm_vec)

    vector_project_vec = project_vector_to_vector(test_vec, norm_vec)
    print("vector_project_vec:", vector_project_vec)

    plane_project_vec = project_vector_to_plane(test_vec, norm_vec)
    print("plane_project_vec:", plane_project_vec)
