import numpy as np


def generate_linear_sweeps(starts, ends, n_points):
    """
    :param starts: An (N, D) array of start points
    :param ends: An (N, D) array of end points
    :param int n_points: The number of points to sweep
    :return: An (N, n_points, D) array of swept points
    """
    frac = np.linspace(0, 1, n_points)[None, :, None]
    sweep = starts[:, None, :]*(1-frac) + ends[:, None, :]*frac
    return sweep


def generate_radial_sweeps(starts, ends, n_points):
    raise NotImplementedError()