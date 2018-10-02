import sys
import argparse
import numpy as np
import torch
import torch.utils.data
from ImageLoader import ImageLoader
from VAE import VAE
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

def loss_function(recon_x, x, mu, logvar):
    MSE = torch.sum((recon_x - x) ** 2)
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE - KLD

def train(epoch, train_dataset, optimiser):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_dataset):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimiser.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss
        optimiser.step()

        if batch_idx % args.log_interval == 0:
            print("Train Epoch {}, batch {} of {}, loss \t{}.".format(
                    epoch,
                    batch_idx,
                    len(train_dataset),
                    loss.data[0]))
    print("Epoch {}, average loss \t\t\t{}".format(
            epoch, 
            train_loss.data[0]/len(train_dataset)))


def test(epoch, test_dataset):
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(test_dataset):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar)
        if batch_idx == 0 and epoch % 10 == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                recon_batch.view(args.batch_size, *size)[:n]])
            save_image(comparison.data.cpu(), 
                       "results/reconstruction_" + str(epoch) + ".png",
                       nrow=n)

    print("Test set loss: \t\t\t{}".format(test_loss.data[0]/len(test_dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE trainer for path planning')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                                help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                                help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=False,
                                help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
    parser.add_argument('--nr-images', type=int, default=10, metavar='N',
                                help='Number of images from the dataset that are used (defaut: 1000)')
    parser.add_argument('--save-path', type=str, default='models/', metavar='P',
                                help='Path to file to save model')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='L',
                                help='The learning rate of the model')
    parser.add_argument('--dataset', type=str, default='images_house/', metavar='D',
                                help='Folder of the data set that contains the images')
    
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda: 
        torch.cuda.manual_seed(args.seed)

    train_images  = 10
    test_images   = 10
    train_dataset = DataLoader(ImageLoader(nr_images=train_images, root_dir=args.dataset, 
                               first_image=1), 
                               batch_size=args.batch_size, 
                               shuffle=True)
    test_dataset  = DataLoader(ImageLoader(nr_images=test_images, root_dir=args.dataset, 
                               first_image=train_images),
                               batch_size=args.batch_size,
                               shuffle=True)
    
    latent_dims = 2
    image_size = (128, 128)
    size = (3, *image_size)
    model = VAE(latent_dims, image_size)
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
            print("Saving model")
            save_file = "{}sistine_chapel_model_epoch_{}_batch_size_{}.pt".format(
                        args.save_path,
                        epoch,
                        args.batch_size)
            torch.save(model, save_file)


