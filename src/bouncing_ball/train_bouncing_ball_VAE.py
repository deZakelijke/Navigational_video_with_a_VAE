import sys
import argparse
import numpy as np
import torch
import torch.utils.data
from BouncingBallLoader import BouncingBallLoader
from simple_VAE import VAE
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

def loss_function(recon_x, x, mu, logvar):
    #MSE = F.mse_loss(recon_x, x, size_average=False)
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE - KLD


def train(epoch, train_dataset, optimiser):
    model.train()
    train_loss = 0 
    for idx, x in enumerate(train_dataset):
        x = Variable(x).float()
        if args.cuda:
            x = x.cuda()
        optimiser.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        train_loss += loss
        optimiser.step()

    print("Training \t epoch: {}, average_loss:\t{}".format(
          epoch,
          train_loss.data[0] / len(train_dataset)))

def test(epoch, test_dataset):
    model.eval()
    test_loss = 0
    for idx, x in enumerate(test_dataset):
        x = Variable(x).float()
        if args.cuda:
            x = x.cuda()
        recon_x , mu, logvar = model(x)
        test_loss += loss_function(recon_x, x, mu, logvar)

        if idx == 0 and epoch % 200 == 0:
            n = min(x.size(0), 8)
            comparison = torch.cat([x[:n],
                recon_x.view(args.batch_size, *size)[:n]])
            save_image(comparison.data.cpu(),
                "results/reconstruction_balls_{}.png".format(epoch),
                nrow=n)
    print("Testing \t epoch: {}, average_loss:\t{}".format(
          epoch,
          test_loss.data[0] / len(test_dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE trainer for bouncing ball data set')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                                help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=False,
                                help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
    parser.add_argument('--nr-images', type=int, default=1000, metavar='N',
                                help='Number of images from the dataset that are used (defaut: 1000)')
    parser.add_argument('--save-path', type=str, default='models/', metavar='P',
                                help='Path to file to save model')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='L',
                                help='The learning rate of the model')
    
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda: 
        torch.cuda.manual_seed(args.seed)

    train_dataset = DataLoader(BouncingBallLoader(n_steps=args.nr_images), 
                               batch_size=args.batch_size, 
                               shuffle=True)
    test_dataset  = DataLoader(BouncingBallLoader(n_steps=args.nr_images // 10),
                               batch_size=args.batch_size,
                               shuffle=True)

    latent_dims = 2
    image_size = (30, 30)
    size = (1, *image_size)
    model = VAE(latent_dims, image_size).float()
    if args.cuda:
        model.cuda()
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        try:
            train(epoch, train_dataset, optimiser)
            test(epoch, test_dataset)

        except KeyboardInterrupt:
            print("Manual quitting")
            sys.exit(0)
        finally:
            if epoch % 100 == 0:
                print("Creating random sample")
                sample = Variable(torch.randn(36, latent_dims))
                if args.cuda:
                    sample = sample.cuda()
                sample = model.decode(sample).cpu()
                save_image(sample.data.view(36, *size), "results/sample_{}.png".format(
                    epoch))

                print("Saving model")
                save_file = "{}bouncing_ball_model_epoch_{}_batch_size_{}.pt".format(
                            args.save_path,
                            epoch,
                            args.batch_size)
                torch.save(model, save_file)


