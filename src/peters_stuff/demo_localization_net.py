import itertools

import torch
import numpy as np
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from scipy.ndimage import gaussian_filter

from artemis.general.numpy_helpers import get_rng


def iter_shifted_blurred_data(patch_size = (20, 20), image_size = (100, 100), blur = 3, rng=None):

    rng = get_rng(rng)
    image_size = np.array(image_size)
    patch_size = np.array(patch_size)
    patch = rng.randn(*patch_size)
    for t in itertools.count(0):
        loc = np.random.rand(2)
        coords = (loc*(image_size-patch_size)).astype(int)
        im = rng.randn(*image_size)
        im[coords[0]:coords[0]+patch_size[0], coords[1]:coords[1]+patch_size[1]] = patch
        blurred_im = gaussian_filter(im, sigma=blur)
        yield loc, blurred_im



def demo_localization_net(patch_size = (20, 20), image_size = (100, 100), n_iter=1000, rng=None):

    rng = get_rng(rng)

    patch = np.random.randn((1, *patch_size))

    for i in range(n_iter):

        position = rng.rand(2)



        im = torch.Tensor()


if __name__ == '__main__':
    for loc, im in iter_shifted_blurred_data():

        crop = (loc * (np.array(im.shape[:2])-(20, 20))).astype(int)

        with hold_dbplots():
            dbplot(im, 'image')
            dbplot(im[crop[0]:crop[0]+20, crop[1]:crop[1]+20], 'crop')

