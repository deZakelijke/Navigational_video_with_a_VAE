import itertools

import torch
from torch import nn
import numpy as np

from artemis.ml.tools.iteration import batchify_generator, generate_in_batches
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from scipy.ndimage import gaussian_filter

from artemis.general.numpy_helpers import get_rng


def iter_shifted_blurred_data(patch_size = (20, 20), image_size = (100, 100), blur = 3, n_iter = None, rng=None):

    rng = get_rng(rng)
    image_size = np.array(image_size)
    patch_size = np.array(patch_size)
    patch = rng.randn(*patch_size)
    for t in itertools.count(0) if n_iter is None else range(n_iter):
        loc = np.random.rand(2)
        coords = (loc*(image_size-patch_size)).astype(int)
        im = rng.randn(*image_size)
        im[coords[0]:coords[0]+patch_size[0], coords[1]:coords[1]+patch_size[1]] = patch
        blurred_im = gaussian_filter(im, sigma=blur)
        yield loc, blurred_im



class CoordConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.coordconv = nn.Conv2d(2, out_channels, kernel_size, stride=stride, padding=padding)
        self.coords = None

    def forward(self, x):
        if self.coords is None:
            sy, sx = x.size()[-2:]
            self.coords = torch.stack(torch.meshgrid([torch.linspace(0, 1, sy), torch.linspace(0, 1, sx)]))[None].to('cuda' if torch.cuda.is_available() else 'cpu')
        return self.conv(x) + self.coordconv(self.coords)


class PatchLocationRegressor(object):

    def __init__(self, kernel_size, ):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.coordconv = nn.Conv2d(2, out_channels, kernel_size, stride=stride, padding=padding)








def demo_localization_net(patch_size = (20, 20), image_size = (100, 100), n_iter=1000, rng=None):

    rng = get_rng(rng)

    patch = np.random.randn((1, *patch_size))

    for loc, im in generate_in_batches(iter_shifted_blurred_data(image_size=image_size, patch_size=patch_size, n_iter=n_iter)):




if __name__ == '__main__':
    for loc, im in iter_shifted_blurred_data():

        crop = (loc * (np.array(im.shape[:2])-(20, 20))).astype(int)

        with hold_dbplots():
            dbplot(im, 'image')
            dbplot(im[crop[0]:crop[0]+20, crop[1]:crop[1]+20], 'crop')

