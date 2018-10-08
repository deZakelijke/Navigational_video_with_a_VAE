import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from bouncing_ball import *

class BouncingBallLoader(Dataset):

    def __init__(self, resolution=30, n_balls=1, n_samples=1, radii=2, n_steps=1600):
        self.data = load_bouncing_ball_data(n_steps=n_steps, resolution=resolution, 
                                            n_balls=n_balls, n_samples=n_samples, 
                                            radii=radii)
        self.size = n_steps

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        return self.data[0,0,index]
