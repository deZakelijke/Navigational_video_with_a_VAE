import random
import string

import numpy as np
import torch

from artemis.fileman.local_dir import get_artemis_data_path
from src.peters_stuff.bbox_utils import bbox_to_position
from src.peters_stuff.image_crop_generator import batch_crop
from src.peters_stuff.pytorch_vae.pytorch_helpers import get_default_device


def chanfirst(im):
    return np.rollaxis(im, 3, 1)


def chanlast(im):
    return np.rollaxis(im, 1, 4)


def normalize_image(im):
    return torch.from_numpy((chanfirst(im)/127.)-1.).float()


def denormalize_image(im):
    return ((chanlast(im.detach().cpu().numpy())+1.)*127.)


def generate_random_model_path(code_gen_len=16, suffix='.pth'):
    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(code_gen_len))
    model_path = get_artemis_data_path('models/{}{}'.format(code, suffix), make_local_dir=True)
    return model_path


def get_normed_crops_and_position_tensors(img, bboxes):
    raw_image_crops = batch_crop(img=img, bboxes=bboxes)
    normed_image_crop_tensors = normalize_image(raw_image_crops).to(get_default_device()).float()
    position_tensors = torch.from_numpy(bbox_to_position(bboxes=bboxes, image_size=img.shape[:2])).to(get_default_device()).float()
    return raw_image_crops, normed_image_crop_tensors, position_tensors


