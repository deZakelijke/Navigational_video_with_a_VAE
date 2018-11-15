import sys
sys.path.append('/home/micha/Documents/repos/Navigational_video_with_a_VAE')
import numpy as np
import torch
from matplotlib import pyplot as plt
from bouncing_balls import *
from src.peters_stuff.sweeps import generate_linear_sweeps
#from simple_VAE import VAE
#from VAE_CoordConv import VAE
from torch.autograd import Variable
from torchvision.utils import save_image
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from artemis.general.image_ops import resize_image


def plot_mapping(old_xy_points, new_xy_points, multi_dims):
    """
    :param old_xy_points: (2xN) array
    :param new_xy_points: (2xN) array
    """

    old_xy_points_norm = (old_xy_points - old_xy_points.min(axis=1, keepdims=True)) / \
                         (old_xy_points.max(axis=1, keepdims=True) - old_xy_points.min(axis=1, keepdims=True))
    colours = [(y, x, 1-x) for x, y in old_xy_points_norm.T]

    ax = plt.subplot(2, 2, 1)
    ax.scatter(old_xy_points[0], old_xy_points[1], c=colours)
    ax.set_title('Ball locations')

    ax = plt.subplot(2, 2, 2)
    ax.scatter(new_xy_points[0], new_xy_points[1], c=colours)
    ax.set_title('Z location (0, 1) with MLP')

    if multi_dims:
        ax = plt.subplot(2, 2, 3)
        ax.scatter(new_xy_points[0], new_xy_points[2], c=colours)
        ax.set_title('Z location (0, 2) with MLP')

        ax = plt.subplot(2, 2, 4)
        ax.scatter(new_xy_points[1], new_xy_points[2], c=colours)
        ax.set_title('Z location (1, 2) with MLP')

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
    print(flat_coordinates_new.shape)
    if flat_coordinates_new.shape[0] > 2:
        multi_dims = True
    else:
        multi_dims = False
    plot_mapping(flat_coordinates_old, flat_coordinates_new, multi_dims)

def plot_latent_variance(model, images, latent_dims):
    images = torch.from_numpy(images).float()
    latent_points = model.encode(images)
    latent_points = (latent_points[0].detach().numpy(), latent_points[1].detach().numpy())
    print(latent_points[1].shape)
    ax = plt.subplot(1, 2, 1)
    ax.plot(latent_points[0].T, "*")
    ax.set_title("Mean")
    ax = plt.subplot(1, 2, 2)
    ax.plot(latent_points[1].T, "*")
    ax.set_title("Logvar")
    plt.show()
    
def random_samples(model, nr_samples, latent_dims, latent_range):
    a = torch.linspace(-latent_range, latent_range, nr_samples)
    b = torch.linspace(-latent_range, latent_range, nr_samples)
    x_t = a.repeat(nr_samples).unsqueeze(1)
    y_t = b.repeat(nr_samples, 1).t().contiguous().view(-1).unsqueeze(1)
    latent_points = torch.cat([x_t, y_t], dim=-1)
    if latent_dims > 2:
        for _ in range(2, latent_dims):
            latent_points = torch.cat([torch.zeros((nr_samples ** 2, 1)), latent_points], dim=-1)
    print(latent_points.shape)
    images = model.decode(latent_points).view(nr_samples ** 2, 1, 30, 30)
    print(images.shape)
    save_image(images, f"results/sample_image_range_minus_{latent_range}_to_{latent_range}.png", nrow=nr_samples)


if __name__ == "__main__":

    RESOLUTION = 30
    N_BALLS = 1
    N_SAMPLES = 1
    RADII = 1.2,
    LATENT_DIMS = 20
    #model_path = "models/TS_bouncing_ball_model.pt"
    model_path = "models/bouncing_ball_model_epoch_1000_batch_size_64.pt"
    model = torch.load(model_path, map_location='cpu')

    nr_points = 1000

    nr_image_samples = 10
    latent_range = 1
    random_samples(model, nr_image_samples, LATENT_DIMS, latent_range)
    images, positions = load_bouncing_ball_data(n_steps=nr_points, resolution=RESOLUTION, n_balls=N_BALLS, n_samples=N_SAMPLES, radii=RADII, save_positions=True)

    #map_images_to_points(model, images, positions)
    plot_latent_variance(model, images, LATENT_DIMS)
