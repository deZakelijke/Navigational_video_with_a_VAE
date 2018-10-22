import sys
print(sys.executable)

from torch.nn import Conv2d

from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.image_ops import resize_image
import torch
import numpy as np
from artemis.plotting.db_plotting import dbplot, dbplot_hang
import numpy as np

def im2feat(im):
    return np.rollaxis(im, -1, -3)


def feat2im(feat):
    return np.rollaxis(feat, -3, feat.ndim)

def manual_cropping_net():
    """
    Can we manually create a network that performs "painting"?

    :return:
    """
    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')

    cut_size = 128
    img = img[img.shape[0]//2-cut_size//2:img.shape[0]//2+cut_size//2, img.shape[1]//2-cut_size//2:img.shape[1]//2+cut_size//2]  # TODO: Revert... this is just to test on a smaller version

    mycrop = img[96:64:-1, 96:64:-1].copy()


    dbplot(img, 'full_img')


    layer = Conv2d(in_channels = 1, out_channels=3, kernel_size=mycrop.shape[:2])

    list(layer.parameters())[0].data = torch.Tensor(im2feat(mycrop)[:, None])

    input_space = torch.zeros((1, 1, 128, 128))

    input_space[0, 0, 40, 60] = 1
    print(input_space.size())
    output = layer(input_space)

    dbplot(feat2im(output.detach().numpy()), 'im')

    dbplot_hang()


if __name__ == "__main__":
    manual_cropping_net()
