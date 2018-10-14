import numpy as np
import torch
from matplotlib import pyplot as plt
from bouncing_balls import *
from simple_VAE import VAE
from torch.autograd import Variable


def plot_mapping(old_xy_points, new_xy_points):
    """
    :param old_xy_points: (2xN) array
    :param new_xy_points: (2xN) array
    """

    old_xy_points_norm = (old_xy_points - old_xy_points.min(axis=1, keepdims=True)) / \
                         (old_xy_points.max(axis=1, keepdims=True) - old_xy_points.min(axis=1, keepdims=True))
    colours = [(y, x, 1-x) for x, y in old_xy_points_norm.T]

    ax = plt.subplot(1, 2, 1)
    ax.scatter(old_xy_points[0], old_xy_points[1], c=colours)
    ax.set_title('Crop Positions')

    ax = plt.subplot(1, 2, 2)
    ax.scatter(new_xy_points[0], new_xy_points[1], c=colours)
    ax.set_title('Z location')
    plt.show()


if __name__ == "__main__":

    RESOLUTION = 30
    N_BALLS = 1
    N_SAMPLES = 1
    RADII = 1.2,
    model_path = "models/bouncing_ball_model_epoch_2000_batch_size_32.pt"
    model = torch.load(model_path)

    nr_points = 10

    images, positions = load_bouncing_ball_data(n_steps=nr_points, resolution=RESOLUTION, n_balls=N_BALLS, n_samples=N_SAMPLES, radii=RADII, save_positions=True)

    flat_coordinates_old = np.array([list(coordinates[0][0]) for coordinates in positions]).T

    model.eval()
    images = torch.from_numpy(images).float().cuda()
    latent_points = model.reparametrise(*model.encode(images)).cpu()

    flat_coordinates_new = [list(coordinates) for coordinates in latent_points.data.numpy()]
    flat_coordinates_new = np.array(flat_coordinates_new).T
    print(flat_coordinates_old)
    print(flat_coordinates_new)
    plot_mapping(flat_coordinates_old, flat_coordinates_new)
