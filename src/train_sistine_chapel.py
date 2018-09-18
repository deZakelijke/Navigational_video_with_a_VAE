import sys
import argparse
import numpy as np
import torch
import torch.utils.data
import ImageLoader
import VAE
from torch import nn, optim
from torch.utils.data import Dataloader
from torchvision import datasets, transforms

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE - KLD

if __name__ == "__main__":
    # argparsing
    parser = argparse.ArgumentParser(description='VAE trainer for path planning')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                                help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=False,
                                help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
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

    
    train_images = 100
    test_images = 10
    train_dataset   = DataLoader(ImageLoader(nr_images=train_images, first_image=0), 
                                 batch_size=args.batch_size, 
                                 shuffle=True)
    test_dataset    = DataLoader(ImageLoader(nr_images=test_images, first_image=train_images),
                                 batch_size=args.batch_size,
                                 shuffle=True)
    
    latent_dims = 2
    image_size = (128, 128)
    size = (3, *image_size)
    model = VAE(latent_dims, image_size)
    if args.cuda():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
