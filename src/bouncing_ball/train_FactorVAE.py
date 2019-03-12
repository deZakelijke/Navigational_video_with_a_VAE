import sys
import argparse
import numpy as np
import torch
import torch.utils.data
from FactorVAE import VAE, Discriminator
from BouncingBallLoader import BouncingBallLoader
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

def get_argparser():
    parser = argparse.ArgumentParser(description="FactorVAE trainer for bouncing ball data set")
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--nr-images', type=int, default=1000, metavar='N',
                        help='Number of images from the dataset that are used (defaut: 1000)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='L',
                        help='The learning rate of the model')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='enable debug mode, saves no models')
    parser.add_argument('--save-path', type=str, default='models/', metavar='S',
                        help='Folder to save the trained models in (default: models/)')
    parser.add_argument('--gamma', type=int, default=40, metavar='G',
                        help='Gamma hyperparameter to regulate training (defaukt: 40)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda: 
        torch.cuda.manual_seed(args.seed)

    return args

def train(vae_model, disc_model, vae_optim, disc_optim, train_dataset, epoch, use_cuda):
    vae_model.train()
    disc_model.train()
    train_loss_VAE = 0
    train_loss_disc = 0
    for idx, minibatch in enumerate(train_dataset):
        labels = torch.FloatTensor(minibatch.shape[0], 1)
        if use_cuda:
            minibatch = minibatch.cuda()
            labels = labels.cuda()
        minibatch = minibatch.float()
# Do all forwards
        recon, mu, logvar, z, z_perm = vae_model(minibatch)
        z_disc = disc_model(z)
        z_disc_perm = disc_model(z_perm)
        #print(sum(z_disc.data), sum(z_disc_perm.data))

# loss for disc
        disc_model.set_grad(True)
        disc_optim.zero_grad()
        labels.data.fill_(1)
        loss = disc_model.loss(z_disc, labels)
        labels.data.fill_(0)
        loss += disc_model.loss(z_disc_perm, labels)
        train_loss_disc += loss
        disc_optim.step()
        disc_model.set_grad(False)

# loss for VAE
        vae_optim.zero_grad()
        loss = vae_model.loss(minibatch, recon, mu, logvar, z_disc)
        train_loss_VAE += loss
        loss.backward()
        vae_optim.step()

    if epoch % 5 == 0:
        print(f"Training>>>> epoch: {epoch}, average loss VAE:\t{train_loss_VAE / len(train_dataset):0.4f}\taverage loss disc:\t{train_loss_disc / len(train_dataset):04f}")
    return train_loss_VAE / len(train_dataset), train_loss_disc / len(train_dataset)


def test(vae_model, disc_model, test_dataset, epoch, size, use_cuda, batch_size):
    vae_model.eval()
    disc_model.eval()
    test_loss = 0

    if epoch % 20:
        return

    test_loss = 0
    for idx, minibatch in enumerate(test_dataset):
        labels = torch.FloatTensor(minibatch.shape[0], 1)
        if use_cuda:
            minibatch = minibatch.cuda()
            labels = labels.cuda()
        minibatch = minibatch.float()
        recon, mu, logvar, z, z_perm = vae_model(minibatch)
        z_disc = disc_model(z)
        z_disc_perm = disc_model(z_perm)

        labels.data.fill_(1)
        loss = disc_model.loss(z_disc, labels)
        labels.data.fill_(0)
        loss += disc_model.loss(z_disc_perm, labels)
        loss += vae_model.loss(minibatch, recon, mu, logvar, z_disc)
        test_loss += loss

       
        if not epoch % 100 and not idx:
            n = min(minibatch.size(0), 8)
            comparison = torch.cat([minibatch[:n],
                recon.view(batch_size, *size)[:n],
                (vae_model.decode(z_perm)).view(batch_size, *size)[:n]])
            save_image(comparison.data.cpu(),
                f"results/reconstruction_FactorVAE_{epoch}.png", nrow=n)

    print(f"Testing>>>>> epoch: {epoch}, average total loss:\t{test_loss/len(test_dataset):0.4f}")
    return test_loss / len(test_dataset)


def run_training_session(use_cuda, gamma, nr_images, learning_rate, epochs, batch_size, save_path, debug=False):
    latent_dims = 20
    image_size = (30, 30)
    size = (1, *image_size)
    use_positions = False
    normalize = True
    train_dataset = DataLoader(BouncingBallLoader(n_steps=nr_images, 
                                                  save_positions=use_positions,
                                                  normalize=normalize), 
                               batch_size=batch_size, 
                               shuffle=True)
    test_dataset  = DataLoader(BouncingBallLoader(n_steps=nr_images // 10, 
                                                  save_positions=use_positions,
                                                  normalize=normalize),
                               batch_size=batch_size,
                               shuffle=True)
    print("Dataset created")
    vae_model = VAE(latent_dims=latent_dims, image_size=image_size, gamma=gamma).float()
    disc_model = Discriminator(latent_dims=latent_dims, image_size=image_size).float()
    if use_cuda and torch.cuda.is_available():
        vae_model = vae_model.cuda()
        disc_model = disc_model.cuda()

    vae_optim = optim.Adam(vae_model.parameters(), lr=learning_rate)
    disc_optim = optim.Adam(disc_model.parameters(), lr=learning_rate)

    try:
        for epoch in range(1, epochs + 1):
            train(vae_model, disc_model, vae_optim, disc_optim, train_dataset, epoch, use_cuda)
            test(vae_model, disc_model, test_dataset, epoch, size, use_cuda, batch_size)
    except KeyboardInterrupt:
        print("Manual interruption of training")
        sys.exit(0)
    finally:
        if not debug:
            print("Saving model")
            save_file_VAE = f"{save_path}bouncing_ball_FactorVAE_vae_epochs_{epoch}_nr_images_{nr_images}_gamma_{gamma}.pt"
            torch.save(vae_model, save_file_VAE)
            save_file_Disc = f"{save_path}bouncing_ball_FactorVAE_disc_epochs_{epoch}_nr_images_{nr_images}_gamma_{gamma}.pt"
            torch.save(disc_model, save_file_Disc)
    return vae_model, disc_model


if __name__ == "__main__":
    args = get_argparser()
    run_training_session(args.cuda, 
                         args.gamma,
                         args.nr_images, 
                         args.learning_rate, 
                         args.epochs, 
                         args.batch_size, 
                         args.save_path,
                         debug=args.debug)

