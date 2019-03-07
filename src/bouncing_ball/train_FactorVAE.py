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

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda: 
        torch.cuda.manual_seed(args.seed)

    return args

def train(vae_model, disc_model, vae_optim, disc_optim, train_dataset, epoch):
    vae_model.train()
    disc_model.train()
    train_loss = 0
    for idx, minibatch in enumerate(train_dataset):
        labels = torch.FloatTensor(minibatch.shape[0], 1)
        if args.cuda:
            minibatch = minibatch.cuda()
            labels = labels.cuda()
        minibatch = minibatch.float()
# Do all forwards
        recon, mu, logvar, z, z_perm = vae_model(minibatch)
        z_disc = disc_model(z)
        z_disc_perm = disc_model(z_perm)
        #print(sum(z_disc.data), sum(z_disc_perm.data))

# loss for disc
        disc_optim.zero_grad()
        labels.data.fill_(1)
        loss = disc_model.loss(z_disc, labels)
        labels.data.fill_(0)
        loss -= disc_model.loss(z_disc_perm, labels)
        train_loss += loss
        loss.backward(retain_graph=True)
        disc_optim.step()

# loss for VAE
        vae_optim.zero_grad()
        loss = vae_model.loss(minibatch, recon, mu, logvar, z_disc)
        train_loss += loss
        loss.backward()
        vae_optim.step()


    print(f"Training>>> epoch: {epoch}, average_loss:\t{train_loss / len(train_dataset)}")

def test(vae_model, disc_model, test_dataset, epoch):
    vae_model.eval()
    disc_model.eval()
    test_loss = 0

    if epoch % 10:
        return

    for idx, minibatch in enumerate(test_dataset):
        labels = torch.FloatTensor(minibatch.shape[0], 1)
        if args.cuda:
            minibatch = minibatch.cuda()
            labels = labels.cuda()
        minibatch = minibatch.float()

# Do all forwards
        recon, mu, logvar, z, z_perm = vae_model(minibatch)
        z_disc = disc_model(z)
        z_disc_perm = disc_model(z_perm)
        #print(sum(z_disc.data), sum(z_disc_perm.data))

# loss for disc
        labels.data.fill_(1)
        loss = disc_model.loss(z_disc, labels)
        labels.data.fill_(0)
        loss -= disc_model.loss(z_disc_perm, labels)
        test_loss += loss

# loss for VAE
        loss = vae_model.loss(minibatch, recon, mu, logvar, z_disc)
        test_loss += loss
        
        if not epoch % 100 and not idx:
            n = min(minibatch.size(0), 8)
            comparison = torch.cat([minibatch[:n],
                recon.view(args.batch_size, *size)[:n],
                (vae_model.decode(z_perm)).view(args.batch_size, *size)[:n]])
            save_image(comparison.data.cpu(),
                f"results/reconstruction_FactorVAE_{epoch}.png", nrow=n)

    print(f"Testing>> epoch: {epoch}, average loss:\t{test_loss/len(test_dataset)}")

# display reconstruction for both permutated and generated



if __name__ == "__main__":
    args = get_argparser()

    latent_dims = 20
    image_size = (30, 30)
    size = (1, *image_size)
    gamma = 40
    use_positions = False
    normalize = True
    train_dataset = DataLoader(BouncingBallLoader(n_steps=args.nr_images, 
                                                  save_positions=use_positions,
                                                  normalize=normalize), 
                               batch_size=args.batch_size, 
                               shuffle=True)
    test_dataset  = DataLoader(BouncingBallLoader(n_steps=args.nr_images // 10, 
                                                  save_positions=use_positions,
                                                  normalize=normalize),
                               batch_size=args.batch_size,
                               shuffle=True)
    print("Dataset created")
    vae_model = VAE(latent_dims=latent_dims, image_size=image_size, gamma=gamma).float()
    disc_model = Discriminator(latent_dims=latent_dims, image_size=image_size).float()
    if args.cuda:
        vae_model = vae_model.cuda()
        disc_model = disc_model.cuda()

    vae_optim = optim.Adam(vae_model.parameters(), lr=args.learning_rate)
    disc_optim = optim.Adam(disc_model.parameters(), lr=args.learning_rate)

    try:
        for epoch in range(1, args.epochs + 1):
            train(vae_model, disc_model, vae_optim, disc_optim, train_dataset, epoch)
            test(vae_model, disc_model, test_dataset, epoch)
    except KeyboardInterrupt:
        print("Manual interruption of training")
        sys.exit(0)
    finally:
        print("Saving model")
        print("Saving not implemented")
