import numpy as np
from matplotlib import pyplot as plt


def get_2d_point_colours(points):
    points_norm = (points - points.min(axis=1, keepdims=True)) / (points.max(axis=1, keepdims=True) - points.min(axis=1, keepdims=True))
    return [(y, x, 1-x) for x, y in points_norm.T]


def plot_mapping(old_xy_points, new_xy_points):
    """
    :param old_xy_points: (2xN) array
    :param new_xy_points: (2xN) array
    """

    colours = get_2d_point_colours(old_xy_points)

    ax = plt.subplot(1, 2, 1)
    ax.scatter(old_xy_points[0], old_xy_points[1], c=colours)
    ax.set_title('Crop Positions')

    ax = plt.subplot(1, 2, 2)
    ax.scatter(new_xy_points[0], new_xy_points[1], c=colours)
    ax.set_title('Z location')
    plt.show()


if __name__ == '__main__':
    n_x = 40
    n_y = 30

    # Generate a grid of points
    old_xy_points = np.array([v.flatten() for v in np.meshgrid(np.linspace(0, 1, n_y), np.linspace(0, 1, n_x))])

    # Apply some transformation
    theta = 5*np.pi/6
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    new_xy_points = np.tanh(transform_matrix @ old_xy_points)

    plot_mapping(old_xy_points, new_xy_points)
