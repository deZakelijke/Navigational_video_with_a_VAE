import sys
import argparse
import numpy as np
import torch
from FactorVAE import VAE, Discriminator
from train_FactorVAE import run_training_session
from bouncing_balls import load_bouncing_ball_data

def calculate_latent_variance(model, images, gamma, use_cuda):
    images = torch.from_numpy(images).float()
    if use_cuda:
        images = images.cuda()
    latent_points = model.encode(images)
    latent_points = (latent_points[0].detach().cpu().numpy(), latent_points[1].detach().cpu().numpy())
    stdevs_of_means = np.std(latent_points[0], axis=0)
    stdevs_of_stds  = np.std(latent_points[1], axis=0)
    nr_active_dimensions_mean = np.sum(stdevs_of_means > 1)
    nr_active_dimensions_std = np.sum(stdevs_of_stds > 1)
    print(f"Gamma value of {gamma}")
    print(f"Number of active dimensions in mean: {nr_active_dimensions_mean}")
    print(f"Number of active dimensions in std: {nr_active_dimensions_std}")


def run_experiments(gamma_range):
    use_cuda = True
    nr_images = 1000
    learning_rate = 1e-4
    epochs = 1500
    batch_size = 32
    save_path = "models/"
    images = load_bouncing_ball_data(n_steps=nr_images, resolution=30, n_samples=1, save_positions=False)

    for gamma in gamma_range:
        print(f"Training with a gamma value of {gamma}")
        vae_model, disc_model = run_training_session(use_cuda,
                                                     gamma,
                                                     nr_images,
                                                     learning_rate,
                                                     epochs,
                                                     batch_size,
                                                     save_path)
        print(f"Analysing latent dimension")
        calculate_latent_variance(vae_model, images, gamma, use_cuda)


if __name__ == '__main__':
    gamma_range = range(5, 50, 5)
    run_experiments(gamma_range)
