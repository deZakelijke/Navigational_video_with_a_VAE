from __future__ import print_function

import numpy as np
import torch
import torch.utils.data
from artemis.experiments import experiment_function
from artemis.ml.tools.iteration import batchify_generator
from artemis.plotting.demo_dbplot import demo_dbplot

from generator import Generator
from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.image_ops import resize_image
#from artemis.general.measuring_periods import measure_period
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from torch import optim
from torch.autograd import Variable

from get_celeb_a import get_celeb_a_iterator
from image_crop_generator import get_image_batch_crop_generator
from sweeps import generate_linear_sweeps

@experiment_function
def demo_train_image_generator(
        batch_size=64,
        cuda=False,
        seed=1234,
        checkpoints={0:10, 100:100, 1000: 1000, 5000:5000},
        learning_rate=1e-3,
        coordinate_dims = 2,
        image_size = (64, 64),
        ):

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    generator_model = Generator(coordinate_dims, image_size)
    if cuda:
        generator_model.cuda()
    optimiser = optim.Adam(generator_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    def train_step(coordinates, images):
        coordinates = Variable(coordinates)
        images = Variable(images)
        if cuda:
            coordinates = coordinates.cuda()
            images = images.cuda()

        generator_model.zero_grad()
        generated_images = generator_model(coordinates)
        loss = generator_model.loss_function(images, generated_images)
        loss.backward()
        optimiser.step()

        return loss

    def convert_to_coordinates(bboxes):
        """ Convert bounding boxes to centroids
            
        Convers the array of boundinx boxes to an array of
        centroids. For simplicity it is assumed that the bounding
        boxes are all 64x64.
        
        Args:
            bboxes (2D numpy array): array of bounding boxes
        """
        coordinates = np.zeros((bboxes.shape[0], 2))
        coordinates[:, 0] = bboxes[:, 0] + 32
        coordinates[:, 1] = bboxes[:, 1] + 32
        return coordinates


    #is_checkpoint = Checkpoints(checkpoints)

    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')

    dbplot(img, 'full_img')

    mode = 'random'
    for i, (bboxes, image_crops) in enumerate(get_image_batch_crop_generator(img=img, crop_size=image_size, batch_size=batch_size, mode=mode, speed=10, randomness=0.1)):

        coordinates = convert_to_coordinates(bboxes)
        coordinates = torch.from_numpy(coordinates).float()

        image_crops = (image_crops.astype(np.float32))/256

        var_image_crops = torch.Tensor(np.rollaxis(image_crops, 3, 1))
        if cuda:
            var_image_crops = var_image_crops.cuda()
        train_loss = train_step(coordinates, var_image_crops)

        if i in checkpoints.values():
            print('Checkping')

            generated_images = generator_model(coordinates)

            with hold_dbplots():
                dbplot(np.rollaxis(generated_images.detach().cpu().numpy(), 1, 4), 'recons')

        break


if __name__ == '__main__':
    demo_train_image_generator(cuda=False)
