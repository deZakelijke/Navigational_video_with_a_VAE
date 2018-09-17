import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils

class ImageLoader(Dataset):

    def __init__(self, nr_images=100, root_dir='images/'):
        self.path = root_dir
        if os.path.isfile(root_dir + str(nr_images + 1) + '.png'):
            self.nr_images = nr_images
        else:
            raise IndexError('Image index does not exist in dir, Dataset not created')

    def __len__(self):
        return self.nr_images

    def __getitem(self, index):
        path = self.path + str(idx + 1) + '.png'
        image = Image.open(path)
        image = image.resize((128, 128))
        image = transforms.ToTensor()(img)
        return image

