from __future__ import print_function

import matplotlib
matplotlib.use('TKAgg')

import sys
sys.path.append('/home/mgroot/Navigational_video_with_a_VAE')
#TODO fix python path
import numpy as np
import torch
from src.VAE_with_Disc import TemporallySmoothVAETrainer
from src.VAE_with_Disc import VAETrainer
#from src.VAE_with_Disc import VAE
from VAE_CoordConv import VAE
from map_latent_ball_points import map_images_to_points
from BouncingBallLoader import BouncingBallLoader
from artemis.experiments import experiment_function
from artemis.ml.tools.iteration import batchify_generator
from artemis.plotting.demo_dbplot import demo_dbplot

from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.image_ops import resize_image

from artemis.plotting.db_plotting import dbplot, hold_dbplots
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


@experiment_function
def demo_train_bouncing_ball_generator(
        batch_size=64,
        cuda=False,
        seed=1234,
        checkpoints={0:10, 50:50, 100:100, 200:200},
        learning_rate=1e-3,
        coordinate_dims = 2,
        image_size = 30,
        n_steps = 16000,
        radii = 1.2,
        save_file = 'results/TS_bouncing_ball_model.pt'
        ):

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        device = 'cuda'
    else:
        deivce = 'cpu'

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    vae = VAE(latent_dims=coordinate_dims, image_size=(image_size, image_size), channels=1)
    if cuda:
        vae = vae.cuda()
#    trainer = TemporallySmoothVAETrainer(vae, learning_rate, device)
    trainer = VAETrainer(vae, learning_rate)
    ball_dataset = DataLoader(BouncingBallLoader(n_steps=n_steps, save_positions=True), 
                                                 batch_size=batch_size, shuffle=False)
    for i, data in enumerate(ball_dataset):
        ball_images = Variable(data[0].float())
        ball_positions = Variable(data[1].float())
        if cuda:
            ball_images = ball_images.cuda()

        train_loss = trainer.train_step(ball_images)

    torch.save(vae, save_file)
#        if i in checkpoints.values():
#            print(f'Checking topologicalness')
#
#            with hold_dbplots():
#                plot_data = ball_positions.detach().cpu().numpy()[:, 0, 0, :]
#                print(plot_data.shape)
#                dbplot(plot_data, 'recons')
    
if __name__ == '__main__':
    demo_train_bouncing_ball_generator(cuda=True, batch_size=32)
