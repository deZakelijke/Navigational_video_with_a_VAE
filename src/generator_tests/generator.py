import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms

class Generator(nn.Module):
    """ Class that can possibly generate images.

    Model is trained in a supervised way with a dataset of random crops from
    a larger image and the coordinates of that crop in tthe image.
    The input signal is the 2D coordinate of the crop and the label is the 
    image that should be generated.

    Args:
        coordinate_dims (int): Number of dimensions of the input coordinate
        image_size (int, int): dimensions of the output image
    """
    def __init__(self, coordinate_dims=2, image_size=(64, 64)):
        super().__init__()

        self.input_dims = coordinate_dims
        self.img_chns = 3
        self.image_size = image_size
        self.filters = 32
        self.flat = 512 * 4

        self.fc_d    = nn.Linear(self.input_dims, self.flat * 4)
        self.bn_d1   = nn.BatchNorm1d(self.flat * 4)
        self.deConv1 = nn.ConvTranspose2d(self.filters * 16, self.filters * 8, 3,
                                          stride=2, padding=0)
        self.bn_d2   = nn.BatchNorm2d(self.filters * 8)
        self.deConv2 = nn.ConvTranspose2d(self.filters * 8, self.filters * 4, 3,
                                          stride=2, padding=1)
        self.bn_d3   = nn.BatchNorm2d(self.filters * 4)
        self.deConv3 = nn.ConvTranspose2d(self.filters * 4, self.filters * 2, 3,
                                          stride=2, padding=1)
        self.bn_d4   = nn.BatchNorm2d(self.filters * 2)
        self.deConv4 = nn.ConvTranspose2d(self.filters * 2, self.filters, 3,
                                          stride=2, padding=1)
        self.bn_d5   = nn.BatchNorm2d(self.filters)
        self.conv_d  = nn.Conv2d(self.filters, self.img_chns, 4, 
                                 stride=1, padding=1)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, coordinate):
        tmp = self.fc_d(coordinate)
        h1 = self.relu(self.bn_d1(tmp))
        h2 = h1.view(-1, self.flat // 4, 4, 4)
        h3 = self.relu(self.bn_d2(self.deConv1(h2)))
        h4 = self.relu(self.bn_d3(self.deConv2(h3)))
        h5 = self.relu(self.bn_d4(self.deConv3(h4)))
        h6 = self.relu(self.bn_d5(self.deConv4(h5)))
        h7 = self.sigmoid(self.conv_d(h6))
        return h7


    def loss_function(self, images, generated_images):
        BCE = F.binary_cross_entropy(generated_images, images, size_average=False)
        return BCE
