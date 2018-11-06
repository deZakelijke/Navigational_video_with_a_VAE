from __future__ import print_function

from argparse import Namespace

import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
import time
from artemis.experiments import experiment_root
from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.image_ops import resize_image
from artemis.ml.tools.iteration import batchify_generator
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from src.gqn.gqn_draw import generator_rnn
from src.gqn.gqn_params import set_gqn_param, get_gqn_param
from src.peters_stuff.image_crop_generator import batch_crop, \
    iter_bboxes_from_positions, iter_pos_random


@experiment_root
def demo_train_just_vae_on_images_gqn(
        batch_size=64,
        checkpoints={0:10, 100:100, 1000: 1000},
        image_size = (64, 64),
        n_iter = None,
        ):

    # latent_dims = 2
    # torch.manual_seed(seed)
    # # if cuda:
    # #     torch.cuda.manual_seed(seed)
    #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # model = VAETrainer(
    #     vae = VAE(latent_dims=latent_dims, image_size=image_size).to(device),
    #     learning_rate=learning_rate,
    # )
    # model = TemporallySmoothVAETrainer(
    #     vae = VAE(latent_dims=latent_dims, image_size=image_size).to(device),
    #     learning_rate=learning_rate,
    #     device=device
    # )

    # model = VAE(latent_dims=latent_dims, image_size=image_size, filters=filters).to(device)
    # opt = torch.optim.Adam(list(model.parameters()), lr = learning_rate, betas = (0.5, 0.999))

    # from src.gqn.gqn_params import PARAMS
    # PARAMS.POSE_CHANNELS = 2

    set_gqn_param('POSE_CHANNELS', 2)

    enc_h, enc_w = get_gqn_param('ENC_HEIGHT'), get_gqn_param('ENC_WIDTH')
    g = Namespace()
    g.positions = tf.placeholder(dtype=tf.float32, shape=(batch_size, 2))
    g.targets = tf.placeholder(dtype=tf.float32, shape=(batch_size, *image_size, 3))
    # g.representations = tf.zeros(dtype=tf.float32, shape=(batch_size, )+image_size+(1, ))
    g.representations = tf.zeros(dtype=tf.float32, shape=(batch_size, enc_h, enc_w, 1))
    # g.query_poses = tf.placeholder(dtype = tf.float32, shape = (batch_size, 2))
    g.mu_targ, _ = generator_rnn(representations=g.representations, query_poses=g.positions, sequence_size=12)
    g.loss = tf.reduce_mean((g.mu_targ-g.targets)**2)
    g.update_op = AdamOptimizer().minimize(g.loss)
    # img = tf.placeholder(dtype=tf.float32, shape=(batch_size, )+image_size+(3, ), name='img')

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    is_checkpoint = Checkpoints(checkpoints)

    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')

    # cut_size = 128
    cut_size = 512
    img = img[img.shape[0]//2-cut_size//2:img.shape[0]//2+cut_size//2, img.shape[1]//2-cut_size//2:img.shape[1]//2+cut_size//2]  # TODO: Revert... this is just to test on a smaller version

    dbplot(img, 'full_img')

    # for i, image_crops in enumerate(get_celeb_a_iterator(minibatch_size=batch_size, size=image_size)):
    # mode = 'smooth'
    mode = 'random'
    t_start = time.time()

    batched_bbox_generator = batchify_generator(list(
        iter_bboxes_from_positions(
            img_size=img.shape[:2],
            crop_size=image_size,
            position_generator=iter_pos_random(n_dim=2, rng=None),
        ) for _ in range(batch_size)))

    for i, bboxes in enumerate(batched_bbox_generator):

        if n_iter is not None and i>=n_iter:
            break

        # dbplot(image_crops, 'crops')
        image_crops = (batch_crop(img=img, bboxes=bboxes))

        image_crops = (image_crops.astype(np.float32))/127.5 - 1.
        positions = np.array(bboxes)[:, [1, 0]] / (img.shape[0]-image_size[0], img.shape[1]-image_size[1]) - 0.5

        # predicted_imgs = sess.run(g.mu_targ, feed_dict={g.positions: positions})
        predicted_imgs, _, loss = sess.run([g.mu_targ, g.update_op, g.loss] , feed_dict={g.positions: positions, g.targets: image_crops})

        if do_every('10s'):
            report = f'Iter: {i}, Loss: {loss:3g}, Mean Rate: {i/(time.time()-t_start):.3g}iter/s'
            print(report)

            with hold_dbplots():
                dbplot(image_crops, 'crops')
                dbplot(predicted_imgs, 'predicted_crops', cornertext=report)

        # loss = tf.reduce_mean((predicted_imgs-image_crops)**2)
        #
        # update_op = AdamOptimizer().minimize(loss)
        #
        # pred_imgs, _ = sess.run([predicted_imgs, update_op], feed_dict={g.query_poses: positions})
        #
        # # var_image_crops = torch.Tensor(np.rollaxis(image_crops, 3, 1)).to(device)
        # # if cuda:
        # #     var_image_crops = var_image_crops.cuda()
        # # vae_loss = model.train_step(var_image_crops)
        #
        #
        # # predicted_imgs = model.decode(torch.Tensor(positions).to(device))
        #
        # # loss = binary_cross_entropy(predicted_imgs, var_image_crops, size_average = False)
        # # loss = torch.nn.functional.mse_loss(predicted_imgs, var_image_crops, size_average = False)
        # # loss = torch.nn.functional.mse_loss(predicted_imgs, var_image_crops, size_average = True)
        # # loss.backward()
        # # opt.step()
        #
        #
        # rate = 1/measure_period('train_step')
        # if do_every('5s'):
        #     print(f'Iter: {i}, Iter/s: {rate:.3g}, Loss: {loss:.3g}')
        #
        # if is_checkpoint():
        #     print('Checking')
        #
        #     # recons, _, _ = model.vae(var_image_crops)
        #
        #     # z_points = torch.randn([batch_size, latent_dims]).float()
        #     # z_grid = torch.Tensor(np.array(np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))).reshape(2, -1).T)
        #
        #     # n_sweep_samples = 8
        #     # n_sweep_points = 8
        #     # z_grid = torch.Tensor(generate_linear_sweeps(starts = np.random.rand(n_sweep_samples, latent_dims), ends=np.random.randn(n_sweep_samples, latent_dims), n_points=n_sweep_points).reshape(n_sweep_points*n_sweep_samples, latent_dims))
        #     # z_grid = torch.Tensor(generate_linear_sweeps(starts = np.zeros((n_sweep_samples, latent_dims))-.5, ends=np.zeros((n_sweep_samples, latent_dims))+.5, n_points=n_sweep_points).reshape(n_sweep_points*n_sweep_samples, latent_dims))
        #
        #     grid_size = 8
        #     z_grid = torch.Tensor(np.array(np.meshgrid(*[[np.linspace(-.5, .5, grid_size)]]*2)).reshape(2, grid_size**2).T)
        #
        #     if cuda:
        #         # z_points = z_points.cuda()
        #         z_grid = z_grid.cuda()
        #
        #     # samples = model.decode(z_points)
        #     grid_samples = model.decode(z_grid)
        #
        #     with hold_dbplots():
        #         dbplot(np.rollaxis(var_image_crops.detach().cpu().numpy(), 1, 4), 'crops')
        #         # dbplot(np.rollaxis(recons.detach().cpu().numpy(), 1, 4), 'recons')
        #         dbplot(np.rollaxis(predicted_imgs.detach().cpu().numpy(), 1, 4), 'predictions', cornertext = f'Iter {i}')
        #         # dbplot(np.rollaxis(samples.detach().cpu().numpy(), 1, 4), 'samples', cornertext = f'Iter {i}')
        #         dbplot(np.rollaxis(grid_samples.detach().cpu().numpy().reshape((grid_size, grid_size, 3)+image_size), 2, 5), 'sweeps', cornertext = f'Iter {i}')
        #         # dbplot(torch.exp(model.transition_logvar), 'transitions', plot_type='line')
        # dbplot(loss, 'loss', plot_type=DBPlotTypes.LINE_HISTORY_RESAMPLED, draw_now=False)



# X32 = demo_train_just_vae_on_images.add_variant(filters = 32, n_iter=5000)
# X64 = demo_train_just_vae_on_images.add_variant(filters = 64, n_iter=5000)
# X128 = demo_train_just_vae_on_images.add_variant(filters = 128, n_iter=5000)

if __name__ == '__main__':
    # X64.run()
    # X32.run()
    # X64.run()
    # X128.run()
    # demo_train_just_vae_on_images.browse()
    demo_train_just_vae_on_images_gqn()
