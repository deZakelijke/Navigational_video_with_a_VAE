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
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE - KLD


def train(epoch, train_dataset, optimiser, use_positions=False):
    model.train()
    train_loss = 0 
    for idx, x in enumerate(train_dataset):
        if use_positions:
            position = x[1]
            position = position.float().view(-1, desired_latent_dims)
            if args.cuda:
                position = position.cuda()
            x = x[0]
        x = Variable(x).float()
        if args.cuda:
            x = x.cuda()
        optimiser.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = model.loss_function_with_SSV(recon_x, x, mu, logvar, position)
        #loss = model.loss_function(recon_x, x, mu, logvar)
        loss.backward()
        train_loss += loss
        optimiser.step()

    print("Training \t epoch: {}, average_loss:\t{}".format(
          epoch,
          train_loss / len(train_dataset)))

def test(epoch, test_dataset, use_positions=False):
    model.eval()
    test_loss = 0
    for idx, x in enumerate(test_dataset):
        if use_positions:
            position = x[1]
            position = position.float().view(-1, desired_latent_dims)
            if args.cuda:
                position = position.cuda()
            x = x[0]
        x = Variable(x).float()
        if args.cuda:
            x = x.cuda()
        recon_x , mu, logvar = model(x)
        test_loss += model.loss_function_with_SSV(recon_x, x, mu, logvar, position)
        #test_loss += model.loss_function(recon_x, x, mu, logvar)

        if idx == 0 and epoch % 200 == 0:
            n = min(x.size(0), 8)
            comparison = torch.cat([x[:n],
                recon_x.view(args.batch_size, *size)[:n]])
            save_image(comparison.data.cpu(),
                "results/reconstruction_balls_{}.png".format(epoch),
                nrow=n)
    print("Testing \t epoch: {}, average_loss:\t{}".format(
          epoch,
          test_loss / len(test_dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE trainer for bouncing ball data set')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                                help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=True,
                                help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
    parser.add_argument('--nr-images', type=int, default=1000, metavar='N',
                                help='Number of images from the dataset that are used (defaut: 1000)')
    parser.add_argument('--save-path', type=str, default='models/', metavar='P',
                                help='Path to file to save model')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='L',
                                help='The learning rate of the model')
    parser.add_argument('--lambda_reg', type=float, default=1.0, metavar='L',
                                help='The regularizing term for the distance loss')
    
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda: 
        torch.cuda.manual_seed(args.seed)

    use_positions = True
    train_dataset = DataLoader(BouncingBallLoader(n_steps=args.nr_images, save_positions=use_positions), 
                               batch_size=args.batch_size, 
                               shuffle=True)
    test_dataset  = DataLoader(BouncingBallLoader(n_steps=args.nr_images // 10, save_positions=use_positions),
                               batch_size=args.batch_size,
                               shuffle=True)

    latent_dims = 20
    desired_latent_dims = 2
    image_size = (30, 30)
    size = (1, *image_size)
    lambda_reg = args.lambda_reg
    model = VAE(latent_dims, image_size, lambda_reg, desired_latent_dims).float()
    if args.cuda:
        model.cuda()
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)

    try:
        for epoch in range(1, args.epochs + 1):
            train(epoch, train_dataset, optimiser, use_positions)
            test(epoch, test_dataset, use_positions)
            if epoch == 1000:
                model.lambda_reg = 0

    except KeyboardInterrupt:
        print("Manual quitting")
        sys.exit(0)
    finally:
        print("Creating random sample")
        sample = Variable(torch.randn(36, latent_dims))
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(36, *size), "results/sample_{}.png".format(
            epoch))

        print("Saving model")
        save_file = f"{args.save_path}bouncing_ball_model_epoch_{epoch}_batch_size_{args.batch_size}_lambda_{args.lambda_reg}.pt"
        torch.save(model, save_file)


