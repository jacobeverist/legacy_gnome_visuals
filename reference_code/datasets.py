
import numpy as np
import pandas as pd
import random

class SignalGenerator(object):
    def __init__(self):
        self.count = 0
        self.val = 0
        self.periodCount = 1

    def __iter__(self):
        self.val = 0
        return self

    def __next__(self):
        self.count += 1

        if self.periodCount % 100 == 0:
            if self.count >= 10:

                self.val += 1
                self.val = self.val % 2
                if self.val == 0:
                    self.periodCount += 1

                self.count = 0
        else:
            if self.count >= 3:

                self.val += 1
                self.val = self.val % 2
                if self.val == 0:
                    self.periodCount += 1

                self.count = 0

        if True:
            return np.float(self.val)
        else:
            raise StopIteration


def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False

    xinters = -1e30

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def make_box_data_grid(h=0.05, max_val=1.1, min_val=-0.1, shuffle=True, rand_seed=2):
    polyRegions = []
    polyRegions.append([(0.1, 0.1), (0.1, 0.4), (0.4, 0.4), (0.4, 0.1)])
    polyRegions.append([(0.6, 0.6), (0.6, 0.9), (0.9, 0.9), (0.9, 0.6)])

    labels = []

    x_min, x_max = min_val, max_val
    y_min, y_max = min_val, max_val
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    xx = xx.ravel()
    yy = yy.ravel()
    zipped_data = list(zip(xx, yy))
    data = np.asarray(zipped_data)
    X = data

    # for k in range(n_samples):
    for k in range(X.shape[0]):
        if point_inside_polygon(X[k, 0], X[k, 1], polyRegions[0]) or point_inside_polygon(X[k, 0], X[k, 1],
                                                                                          polyRegions[1]):
            # inside
            labels.append(0)
        else:
            # outside
            labels.append(1)

    y = np.asarray(labels)

    if shuffle:
        rng = np.random.RandomState(rand_seed)
        rng.shuffle(X)
        rng = np.random.RandomState(rand_seed)
        rng.shuffle(y)

    return X, y


def make_box_data_random(n_samples=500, max_val=1.1, min_val=-0.1, stratify=False, shuffle=True, rand_seed=2):
    polyRegions = []
    polyRegions.append([(0.1, 0.1), (0.1, 0.4), (0.4, 0.4), (0.4, 0.1)])
    polyRegions.append([(0.6, 0.6), (0.6, 0.9), (0.9, 0.9), (0.9, 0.6)])

    labels = []

    # create balanced set of inside/outside samples

    if stratify:
        n_in_samples = n_samples / 2
        n_out_samples = n_samples - n_in_samples

        rng = np.random.RandomState(rand_seed)

        in_samples = []
        out_samples = []

        while len(out_samples) < n_out_samples or len(in_samples) < n_in_samples:
            X = rng.uniform(low=min_val, high=max_val, size=2)

            if point_inside_polygon(X[0], X[1], polyRegions[0]) \
                    or point_inside_polygon(X[0], X[1], polyRegions[1]):
                # inside
                if len(in_samples) < n_in_samples:
                    in_samples.append(X)
            else:
                # outside
                if len(out_samples) < n_out_samples:
                    out_samples.append(X)

        in_labels = [0 for k in range(len(in_samples))]
        out_labels = [1 for k in range(len(out_samples))]
        labels = in_labels + out_labels
        y = np.asarray(labels)

        all_samples = in_samples + out_samples
        X = np.asarray(all_samples)

    else:
        rng = np.random.RandomState(rand_seed)
        X = rng.uniform(low=min_val, high=max_val, size=(n_samples, 2))

        # for k in range(n_samples):
        for k in range(X.shape[0]):
            if point_inside_polygon(X[k, 0], X[k, 1], polyRegions[0]) \
                    or point_inside_polygon(X[k, 0], X[k, 1], polyRegions[1]):
                # inside
                labels.append(0)
            else:
                # outside
                labels.append(1)

        y = np.asarray(labels)

    if shuffle:
        rng = np.random.RandomState(rand_seed)
        rng.shuffle(X)
        rng = np.random.RandomState(rand_seed)
        rng.shuffle(y)

    return X, y


def Clifford(x, y, a, b, c, d, *o):
    return np.sin(a * y) + c * np.cos(a * x), \
           np.sin(b * x) + d * np.cos(b * y)
    # return x + (np.sin(a * y) + c * np.cos(a * x)) * 0.1, \
    #       y + (np.sin(b * x) + d * np.cos(b * y)) * 0.1

    #    xs[i + 1] = xs[i] + (x_dot * dt)
    #    ys[i + 1] = ys[i] + (y_dot * dt)


def trajectory_coords(fn, x0, y0, a, b=0., c=0., d=0., e=0., f=0., n=1000):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n - 1):
        x[i + 1], y[i + 1] = fn(x[i], y[i], a, b, c, d, e, f)
    return x, y


def lemiscate(n=100, alpha=1.0):

    #t = np.linspace(0, 2*np.pi, num=n)
    t = np.linspace(np.pi/2, 2*np.pi+np.pi/2, num=n)

    x = alpha * np.sqrt(2) * np.cos(t) / (np.sin(t)**2 + 1)
    y = alpha * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)

    return x, y


def circle(n=100, r=1.0):
    angles = np.linspace(0, 2 * np.pi, num=n)
    xs = r * np.cos(angles)
    ys = r * np.sin(angles)

    #x = []
    #y = []
    # for theta in np.linspace(0, 10 * np.pi):
    #    r = ((theta) ** 2)
    #    x.append(r * np.cos(theta))
    #    y.append(r * np.sin(theta))

    return xs, ys


def spiral(n=100):
    angles = np.linspace(0, 10 * np.pi, num=n)
    xs = 0.002 * (angles ** 2) * np.cos(angles)
    ys = 0.002 * (angles ** 2) * np.sin(angles)

    #x = []
    #y = []
    # for theta in np.linspace(0, 10 * np.pi):
    #    r = ((theta) ** 2)
    #    x.append(r * np.cos(theta))
    #    y.append(r * np.sin(theta))

    return xs, ys


def random_walk():
    random.seed(6)
    n = 10000  # Number of points
    f = filter_width = 5000  # momentum or smoothing parameter, for a moving average filter

    # filtered random walk
    xs = np.convolve(np.random.normal(0, 0.1, size=n), np.ones(f) / f).cumsum()
    ys = np.convolve(np.random.normal(0, 0.1, size=n), np.ones(f) / f).cumsum()

    # Add "mechanical" wobble on the x axis
    # xs += 0.1 * np.sin(0.1 * np.array(range(n - 1 + f)))

    # Add "measurement" noise
    # xs += np.random.normal(0, 0.005, size=n - 1 + f)
    # ys += np.random.normal(0, 0.005, size=n - 1 + f)

    # Add a completely incorrect value
    # xs[int(len(xs) / 2)] = 100
    # ys[int(len(xs) / 2)] = 0

    # Create a dataframe
    df = pd.DataFrame(dict(x=xs, y=ys))

    return df


def trajectory(fn, x0, y0, a, b=0., c=0., d=0., e=0., f=0., n=1000):
    x, y = trajectory_coords(fn, x0, y0, a, b, c, d, e, f, n)
    return pd.DataFrame(dict(x=x, y=y))


def lorenz(x, y, s=10, r=28, b=2.667):
    '''
    Given:
       x, y: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot: values of the lorenz attractor's partial
           derivatives at the point x, y
    '''

    return
    # x_dot = s * (y - x)
    # y_dot = r * x - y - x * z
    # z_dot = x * y - b * z
    # return x_dot, y_dot, z_dot


# def trajectory(x, y, a=0.1, b=0.1, c=0.1, d=0.1):
#    xnew = np.sin(y * b) + c * np.sin(x * b)
#    ynew = np.sin(x * a) + d * np.sin(y * a)

#    return xnew, ynew


# make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.8)
# make_friedman2(n_samples=100, noise=0.0, random_state=None)
# make_gaussian_quantiles(mean=None, cov=1.0, n_samples=100, n_features=2, n_classes=3, shuffle=True, random_state=None)

# Return
# X : array of shape [n_samples, n_features]
# The generated samples.
#
# y : array of shape [n_samples]
# The integer labels for membership of each sample.