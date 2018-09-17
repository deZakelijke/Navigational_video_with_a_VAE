import sys
import argparse
import numpy as np
import torch
import torch.utils.data
import ImageLoader
import VAE
from torch import nn, optim
from torchvision import datasets, transforms


if not torch.cuda.is_available():
    print("Cuda not available, terminating")
    sys.exit(0)

torch.cuda.manual_seed(1)

nr_images = 100
dataset = ImageLoader(nr_images=nr_images)

learning_rate = 1e-3
latent_dims = 2
image_size = (128, 128)
size = (3, *image_size)
model = VAE(latent_dims, image_size)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
