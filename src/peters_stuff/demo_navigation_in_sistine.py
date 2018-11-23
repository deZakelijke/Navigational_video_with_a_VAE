from collections import namedtuple
from functools import partial
from typing import Callable, Iterator, Tuple, Union

import numpy as np
from scipy.signal import correlate2d

from artemis.fileman.local_dir import get_artemis_data_path
from artemis.general.mymath import argmaxnd
from artemis.plotting.db_plotting import dbplot, hold_dbplots, DBPlotTypes
from artemis.plotting.matplotlib_backend import LinePlot
from src.peters_stuff.bbox_utils import crop_img_with_bbox, clip_bbox_to_image, shift_bbox, get_bbox_overlap
from src.peters_stuff.crop_predictors import ICropPredictor
from src.peters_stuff.image_crop_generator import iter_bbox_batches
from src.peters_stuff.position_predictors import IPositionPredictor, bbox_to_position
from src.peters_stuff.sample_data import SampleImages
from src.peters_stuff.tf_helpers import TFGraphClass, replicate_subgraph, load_model_and_graph
from src.peters_stuff.tf_vaes import TFVAEModel, VAEGraph
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge


class INavigationModel(object):

    def plan_route(self, start_img, dest_img, slice_video=None):
        route, zs = self.plan_route_and_get_zs(start_img, dest_img, slice_video=slice_video)
        return route

    def plan_route_and_get_zs(self, start_img, dest_img, slice_video=None):
        raise NotImplementedError()


class PretrainedStrightLineNavModel(INavigationModel):

    def __init__(self, encoder: IPositionPredictor, decoder: ICropPredictor, step_spacing = 0.05):
        self.encoder = encoder
        self.decoder = decoder
        self.step_spacing = step_spacing

    def plan_route_and_get_zs(self, start_img, dest_img, slice_video = None):
        zs = self.encoder.predict(start_img[None])
        zd = self.encoder.predict(dest_img[None])
        # dbplot(self.decoder.predict(zs), 'start_recon')
        # dbplot(self.decoder.predict(zd), 'dest_recon')

        n_waypoints = max(2, int(np.ceil(np.sqrt(((zs-zd)**2).sum()) / self.step_spacing)))
        frac = np.linspace(0, 1, n_waypoints)[:, None]

        zp = zs*(1-frac) + zd*frac

        # print(f'zs={zs}, zd={zd}')
        video = self.decoder.predict(zp if slice_video is None else zp[slice_video])
        return video, zp


class VAEStraightLineNavModel(INavigationModel):

    def __init__(self, vae: TFVAEModel, step_spacing = 0.05):
        self.vae = vae
        self.step_spacing = step_spacing

    def plan_route_and_get_zs(self, start_img, dest_img, slice_video=None):
        start_img = (start_img.astype(np.float32))[None]/255.999
        dest_img = (dest_img.astype(np.float32))[None]/255.999
        zs, _ = self.vae.encode(start_img)
        zd, _ = self.vae.encode(dest_img)
        n_waypoints = max(2, int(np.ceil(np.sqrt(((zs-zd)**2).sum()) / self.step_spacing)))
        frac = np.linspace(0, 1, n_waypoints)[:, None]
        zp = zs*(1-frac) + zd*frac
        video = self.vae.decode(zp)
        return video, zp


PathPlanningNodes = namedtuple('PathPlanningNodes', ['xs', 'xd', 'z_path', 'x_path', 'batch_size'])


class VAEStraightLinePlanner(TFGraphClass[PathPlanningNodes], INavigationModel):
    # TODO: Make this work... It's broken now because of the replicate subgraph thing.

    def plan_route_and_get_zs(self, start_img, dest_img, slice_video=None):
        xs = (start_img.astype(np.float32))[None]/255.999
        xd = (start_img.astype(np.float32))[None]/255.999
        video, zp = self.sess.run([self.nodes.x_path, self.nodes.z_path], feed_dict={self.nodes.xs: xs, self.nodes.xd: xd, self.nodes.batch_size: len(start_img)})
        return video, zp

    @staticmethod
    def from_vae(vae: TFGraphClass[VAEGraph], step_spacing = 0.05):

        nodes = vae.nodes  # type: VAEGraph

        # batch_size = tf.placeholder(tf.int32, [], name='the_batch_size')
        (xs, batch_size), zs = replicate_subgraph(inputs=[nodes.x_sample, nodes.batch_size], outputs=nodes.z_mu, new_inputs={nodes.batch_size: None})
        (xd, batch_size), zd = replicate_subgraph(inputs=[nodes.x_sample, nodes.batch_size], outputs=nodes.z_mu, new_inputs={nodes.batch_size: None})
        n_waypoints = tf.maximum(2, tf.to_int32(tf.ceil(tf.reduce_sum(tf.sqrt(((zs-zd)**2)) / step_spacing))))
        frac = tf.linspace(0., 1., n_waypoints)[:, None]
        zp = zs*(1-frac) + zd*frac

        _, video = replicate_subgraph(inputs=[nodes.z_sample, nodes.batch_size], new_inputs=[zp, batch_size], outputs=vae.nodes.x_mu)

        nodes = PathPlanningNodes(xs=xs, xd=xd, z_path=zp, x_path=video, batch_size=batch_size)
        return VAEStraightLinePlanner(nodes)


def demo_show_navigation_video(
        model,
        position_generator_constructor: Union[str, Callable[[], Iterator[Tuple[int, int]]]] = 'random',
        crop_size = (64, 64),
        ):

    img = SampleImages.sistine_512()
    for i, (start_bbox, dest_bbox) in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=2, position_generator_constructor=position_generator_constructor)):

        start_image = crop_img_with_bbox(img, bbox = start_bbox, crop_edge_setting='error')
        dest_image = crop_img_with_bbox(img, bbox = dest_bbox, crop_edge_setting='error')
        route_video, zs = model.plan_route_and_get_zs(start_image, dest_image)
        true_positions = bbox_to_position((start_bbox, dest_bbox), image_size=img.shape[:2], crop_size=start_image.shape[:2])
        for p_true, p_enc in zip(true_positions, zs[[0, -1]]):
            print(f'{p_true} -> {p_enc}')
        with hold_dbplots():
            dbplot(start_image, 'start')
            dbplot(dest_image, 'dest')
            dbplot(img, 'full_img')
            dbplot(start_bbox, 'start_bbox', axis='full_img', plot_type = DBPlotTypes.BBOX_G)
            dbplot(dest_bbox, 'dest_bbox', axis='full_img', plot_type = DBPlotTypes.BBOX_R)
            dbplot((zs[:, 1], -zs[:, 0]), 'latent_plan', plot_type=partial(LinePlot, add_end_markers=True, x_bounds = (-1.2, 1.2), y_bounds = (-1.2, 1.2)))
        for t, im in enumerate(route_video):
            with hold_dbplots(hang=1. if t in (0, len(route_video)-1) else None):
                dbplot(im, 'route', title=f'Route - Frame {t+1}/{len(route_video)}')
                dbplot((zs[t:, 1], -zs[t:, 0]), 'current_plan', axis='latent_plan', plot_type=partial(LinePlot, add_end_markers=True, x_bounds = (-1.2, 1.2), y_bounds = (-1.2, 1.2), color='r'))


class CrossCorrelationController(object):

    def __init__(self, boundary_mode = 'symm'):
        self.boundary_mode = boundary_mode

    def __call__(self, current_frame, next_frame):
        current_frame = current_frame.astype(float) - current_frame.mean()
        next_frame = next_frame.astype(float) - next_frame.mean()
        corr_img = np.array([correlate2d(current_frame[:, :, i].astype(float), next_frame[:, :, i].astype(float), mode='same', boundary=self.boundary_mode) for i in range(3)]).mean(axis=0)
        maxcorr = argmaxnd(corr_img)
        # dbplot([current_frame, next_frame], 'frames')
        # dbplot(corr_img, 'corr')
        relcorr_y, relcorr_x = maxcorr - np.array(current_frame.shape[:2])/2.
        return (relcorr_x, relcorr_y)


def demo_use_system_controller(
        model: INavigationModel,
        position_generator_constructor: Union[str, Callable[[], Iterator[Tuple[int, int]]]] = 'random',
        crop_size = (64, 64),
        boundary_mode = 'symm',
        timeout = 50,
        skipto=0,
        seed = 1234
        ):

    img = SampleImages.sistine_512()
    controller = CrossCorrelationController(boundary_mode=boundary_mode)
    for i, (start_bbox, dest_bbox) in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=2, position_generator_constructor=position_generator_constructor, rng=seed)):

        if i<skipto:
            continue
        start_image = crop_img_with_bbox(img, bbox = start_bbox, crop_edge_setting='error')
        dest_image = crop_img_with_bbox(img, bbox = dest_bbox, crop_edge_setting='error')

        with hold_dbplots():
            dbplot(start_image, 'start')
            dbplot(dest_image, 'dest')
            dbplot(img, 'full_img')
            dbplot(start_bbox, 'current', axis='full_img', plot_type = DBPlotTypes.BBOX_B)
            dbplot(start_bbox, 'start_bbox', axis='full_img', plot_type = DBPlotTypes.BBOX_G)
            dbplot(dest_bbox, 'dest_bbox', axis='full_img', plot_type = DBPlotTypes.BBOX_R)

        current_bbox = start_bbox
        current_image = start_image
        for t in range(timeout):
            route_video, zs = model.plan_route_and_get_zs(current_image, dest_image, slice_video=slice(None, 4))
            shift = controller(route_video[0], route_video[1])
            last_bbox = current_bbox
            current_bbox = clip_bbox_to_image(shift_bbox(current_bbox, shift), width=img.shape[1], height=img.shape[0], preserve_size=True)
            current_image = crop_img_with_bbox(img, bbox = current_bbox, crop_edge_setting='error')

            with hold_dbplots():
                dbplot(current_image, 'current_image', cornertext=f'Trial {i}, Time {t+1}')
                dbplot(route_video, 'route_plan')
                dbplot(current_bbox, 'current', axis='full_img', plot_type = DBPlotTypes.BBOX_B)
                dbplot((zs[:, 1], -zs[:, 0]), 'latent_plan', title = f'Latent Plan: {len(zs)} steps.', plot_type=partial(LinePlot, add_end_markers=True, x_bounds = (-1.2, 1.2), y_bounds = (-1.2, 1.2)))

            overlap = get_bbox_overlap(current_bbox, dest_bbox)
            print(f'Shift: {shift}, old_bbox: {last_bbox}, new_bbox = {current_bbox}, dest_bbox = {dest_bbox}, Overlap = {overlap:.2g} ')

            if overlap>0.7:
                print(f'Reached Destination in {t} steps.')
                break


if __name__ == "__main__":

    class ImageNavigationDemos:
        """
        To run these experiments, first copy the pretrained models over in terminal with:
            rsync -a petered@146.50.28.7:~/.artemis/tests/models/ ~/.artemis/tests/models/
        """

        @staticmethod
        def pretrained_vid():
            encoder = TFGraphClass.load(get_artemis_data_path('tests/models/ROTBKQWX3DHT30D2/model'))
            decoder = TFGraphClass.load(get_artemis_data_path('tests/models/Y1UDTD7POJ6N0ILD/model'))  # Big one trained overnight
            demo_show_navigation_video(model=PretrainedStrightLineNavModel(encoder=encoder, decoder=decoder, step_spacing=0.01))

        @staticmethod
        def pretrained_tracker():
            encoder = TFGraphClass.load(get_artemis_data_path('tests/models/ROTBKQWX3DHT30D2/model'))
            decoder = TFGraphClass.load(get_artemis_data_path('tests/models/Y1UDTD7POJ6N0ILD/model'))  # Big one trained overnight
            demo_use_system_controller(model=PretrainedStrightLineNavModel(encoder=encoder, decoder=decoder, step_spacing=0.03), boundary_mode='fill')

        @staticmethod
        def vae_tracker():
            # vae = TFGraphClass.load(get_artemis_data_path('tests/models/M0B6AQVS4DF91940/model'))
            # demo_use_system_controller(model = VAEStraightLineNavModel(vae))
            vae = TFGraphClass.load(get_artemis_data_path('tests/models/M0B6AQVS4DF91940/model'))
            demo_use_system_controller(model = VAEStraightLineNavModel(vae))

        @staticmethod
        def vae_planner():
            # vae = TFGraphClass.load(get_artemis_data_path('tests/models/M0B6AQVS4DF91940/model'))
            # demo_use_system_controller(model = VAEStraightLineNavModel(vae))
            vae = TFGraphClass.load(get_artemis_data_path('tests/models/M0B6AQVS4DF91940/model'))

            demo_use_system_controller(model = VAEStraightLinePlanner.from_vae(vae))

    ImageNavigationDemos.vae_planner()
