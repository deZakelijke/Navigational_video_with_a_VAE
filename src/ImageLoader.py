import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils

class ImageLoader(Dataset):

    def __init__(self, nr_images=100, root_dir='images/', first_image=0):
        self.path = root_dir
        if not os.path.isfile(root_dir + str(first_image) + '.png'):
            raise IndexError('Image index does not exist in dir, dataset not created')
        if not os.path.isfile(root_dir + str(nr_images + first_image - 1) + '.png'):
            raise IndexError('Dir does not contain enough images to create dataset, dataset not created')

        self.nr_images = nr_images
        self.first_image = first_image

    def __len__(self):
        return self.nr_images

    def __getitem__(self, index):
        path = self.path + str(index + self.first_image) + '.png'
        image = Image.open(path)
        image = image.resize((128, 128))
        image = transforms.ToTensor()(image)
        return image

