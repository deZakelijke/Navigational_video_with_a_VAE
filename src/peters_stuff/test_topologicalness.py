
import numpy as np
import matplotlib.pyplot as plt
from src.generate_remapping_plots import get_2d_point_colours
from src.peters_stuff.topologicalness import topologicalness


def demo_untopologicalness(k='all'):

    n_rows = 40
    n_cols = 30

    # Generate a grid of points
    x = np.array([v.flatten() for v in np.meshgrid(np.linspace(0, 1, n_cols), np.linspace(0, 1, n_rows))])

    # Transform 1: Linear
    theta = 5*np.pi/6
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    y1 = transform_matrix @ x

    # Transform 2: Nonlinear but topological
    y2 = np.tanh(transform_matrix @ x)

    # Transform 3: Smooth but non-topological
    y3 = np.cos(transform_matrix @ x * 5)
    #
    # Transform 4: Random and totally non-topological
    y4 = np.random.randn(*x.shape)

    colours = get_2d_point_colours(x)

    ax = plt.subplot(1, 5, 1)
    ax.scatter(x[0], x[1], c=colours)
    ax.set_title('x')

    # Apply some transformation
    for i, (name, y) in enumerate([('linear', y1), ('nonlinear', y2), ('overlapping', y3), ('random', y4)]):
        d = topologicalness(x.T, y.T, k=k)

        ax = plt.subplot(1, 5, 2+i)
        ax.scatter(y[0], y[1], c=colours)

        ax.set_title(f'{name}: {d:.3g}')

    plt.show()


if __name__ == "__main__":
    demo_untopologicalness()

