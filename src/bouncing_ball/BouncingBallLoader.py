import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from bouncing_balls import *

class BouncingBallLoader(Dataset):

    def __init__(self, resolution=30, n_balls=1, n_samples=1, radii=2, n_steps=1600, save_positions=False):
        self.data = load_bouncing_ball_data(n_steps=n_steps, resolution=resolution, 
                                            n_balls=n_balls, n_samples=n_samples, 
                                            radii=radii, save_positions=save_positions)
        self.size = n_steps
        self.save_positions=save_positions

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        if self.save_positions:
            return (self.data[0][index], self.data[1][index])
        else:
            return self.data[index]
