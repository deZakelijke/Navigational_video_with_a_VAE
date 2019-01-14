from abc import abstractmethod
from typing import Callable, Tuple

import numpy as np


class ICropPredictor(object):

    @abstractmethod
    def train(self, positions, image_crops):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, positions):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_constructor(**kwargs) -> Callable[[int, Tuple[int,int]], 'ICropPredictor']:
        raise NotImplementedError()


def imbatch_to_feat(im, channel_first, datarange):

    if channel_first:
        im = np.rollaxis(im, 3, 1)
    dmin, dmax = datarange
    feat = (im.astype(np.float32)/255.999)*(dmax-dmin)+dmin
    return feat


def feat_to_imbatch(feat, channel_first, datarange):
    if channel_first:
        feat = np.rollaxis(feat, 1, 4)
    dmin, dmax = datarange
    feat = (((feat-dmin)/(dmax-dmin))*255.999).astype(np.uint8)
    return feat


