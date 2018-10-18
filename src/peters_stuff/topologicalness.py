import numpy as np


def topologicalness(x, y, k='all', distance_measure='L2'):
    """
    Given points x, y, where y is produced by some mapping f: X->Y for points in x, evaluate how
    topological this mapping is.  This returns a lower number the more topological the mapping is.

    :param x: An (N, DX) array of points in X space
    :param y: An (N, DY) array of points in Y space
    :param k: The number of neighbours to use.  Or 'all' to do a weighted combination.
    :param distance_measure: Either a string defining a distance measure (e.g. 'L2') or a function of the form
        d = f(x1, x2)  where x1 and x2 are broadcastable arrays of points and f returns the distance between each pair
            along the last axis
    :return float: A positive measure of "topologicalness" of the mapping.  This will be 1 for a linear mapping, and
        approach 0 for a totally untoploligical mapping.
    """

    n = len(x)
    assert n==len(y), "x, y must have same length"
    if distance_measure=='L2':
        distance_measure = lambda x1, x2: np.sqrt(((x1-x2)**2).sum(axis=-1))
    else:
        assert callable(distance_measure), "distance_measure Must be callable or a pre-defined string"
    dmatx = distance_measure(x[:, None, :], x[None, :, :])
    dmaty = distance_measure(y[:, None, :], y[None, :, :])

    if k=='all':
        sorted_neighbours = np.argsort(dmatx, axis=1)[:, 1:]
        ks = np.arange(1, n)
        ratio_x = (n * dmatx[np.arange(n)[:, None], sorted_neighbours].cumsum(axis=1)) / (ks* dmatx.sum(axis=1, keepdims=True))
        ratio_y = (n * dmaty[np.arange(n)[:, None], sorted_neighbours].cumsum(axis=1)) / (ks* dmaty.sum(axis=1, keepdims=True))
        weights = 1./np.arange(1, n)
        weights = weights/np.sum(weights)
        return (weights * (ratio_x.sum(axis=0) / ratio_y.sum(axis=0))).sum()
    else:
        sorted_neighbours = np.argsort(dmatx, axis=1)
        ratio_y = (n * dmaty[np.arange(n)[:, None], sorted_neighbours[:, :k]].sum(axis=1)) / (k * dmaty.sum(axis=1))
        ratio_x = (n * dmatx[np.arange(n)[:, None], sorted_neighbours[:, :k]].sum(axis=1)) / (k * dmatx.sum(axis=1))
        return ratio_x.sum()/ratio_y.sum()
