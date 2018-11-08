import sys
sys.path.append('/home/micha/Documents/repos/Navigational_video_with_a_VAE')
import numpy as np
import torch
from matplotlib import pyplot as plt
from bouncing_balls import *
from src.peters_stuff.sweeps import generate_linear_sweeps
from simple_VAE import VAE
from torch.autograd import Variable
from torchvision.utils import save_image
from artemis.plotting.db_plotting import dbplot, hold_dbplots


def plot_mapping(old_xy_points, new_xy_points, multi_dims):
    """
    :param old_xy_points: (2xN) array
    :param new_xy_points: (2xN) array
    """

    old_xy_points_norm = (old_xy_points - old_xy_points.min(axis=1, keepdims=True)) / \
                         (old_xy_points.max(axis=1, keepdims=True) - old_xy_points.min(axis=1, keepdims=True))
    colours = [(y, x, 1-x) for x, y in old_xy_points_norm.T]

    ax = plt.subplot(2, 3, 1)
    ax.scatter(old_xy_points[0], old_xy_points[1], c=colours)
    ax.set_title('Ball locations')

    ax = plt.subplot(2, 3, 2)
    ax.scatter(new_xy_points[0], new_xy_points[1], c=colours)
    ax.set_title('Z location (0, 1) with CoordinateConvolution')

    if multi_dims:
        ax = plt.subplot(2, 3, 3)
        ax.scatter(new_xy_points[0], new_xy_points[2], c=colours)
        ax.set_title('Z location (0, 2) with CoordinateConvolution')

        ax = plt.subplot(2, 3, 4)
        ax.scatter(new_xy_points[1], new_xy_points[2], c=colours)
        ax.set_title('Z location (1, 2) with CoordinateConvolution')

        #ax = plt.subplot(2, 3, 5)
        #ax.scatter(new_xy_points[0], new_xy_points[3], c=colours)
        #ax.set_title('Z location (0, 3) with CoordinateConvolution')

        #ax = plt.subplot(2, 3, 6)
        #ax.scatter(new_xy_points[2], new_xy_points[3], c=colours)
        #ax.set_title('Z location (2, 3) with CoordinateConvolution')


    plt.show()

def map_images_to_points(model, images, positions):
    flat_coordinates_old = np.array([list(coordinates[0][0]) for coordinates in positions]).T

    model.eval()
    images = torch.from_numpy(images).float()
    latent_points = model.reparametrize(*model.encode(images))

    flat_coordinates_new = [list(coordinates) for coordinates in latent_points.data.numpy()]
    flat_coordinates_new = np.array(flat_coordinates_new).T
    if flat_coordinates_new.shape[0] > 2:
        multi_dims = True
    else:
        multi_dims = False
    plot_mapping(flat_coordinates_old, flat_coordinates_new, multi_dims)

    
def random_samples(model, nr_samples, latent_dims):
    a = torch.linspace(-1, 1, nr_samples)
    b = torch.linspace(-1, 1, nr_samples)
    x_t = a.repeat(nr_samples).unsqueeze(1)
    y_t = b.repeat(nr_samples, 1).t().contiguous().view(-1).unsqueeze(1)
    latent_points = torch.cat([x_t, y_t], dim=-1)
    print(latent_points.shape)
    images = model.decode(latent_points)
    save_image(images, "results/sample_image.png")

if __name__ == "__main__":

    RESOLUTION = 30
    N_BALLS = 1
    N_SAMPLES = 1
    RADII = 1.2,
    LATENT_DIMS = 3
    model_path = "models/TS_bouncing_ball_model.pt"
    #model_path = "models/bouncing_ball_model_epoch_2000_batch_size_32.pt"
    model = torch.load(model_path, map_location='cpu')

    nr_points = 1000

    random_samples(model, 6, LATENT_DIMS)
    images, positions = load_bouncing_ball_data(n_steps=nr_points, resolution=RESOLUTION, n_balls=N_BALLS, n_samples=N_SAMPLES, radii=RADII, save_positions=True)

    map_images_to_points(model, images, positions)
