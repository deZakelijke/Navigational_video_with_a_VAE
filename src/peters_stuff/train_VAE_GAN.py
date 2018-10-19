from __future__ import print_function

import numpy as np
import torch
import torch.utils.data
from artemis.experiments import experiment_function
from artemis.ml.tools.iteration import batchify_generator
from artemis.plotting.demo_dbplot import demo_dbplot

from src.VAE_with_Disc import VAE, Discriminator
from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.image_ops import resize_image
from artemis.general.measuring_periods import measure_period
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from torch import optim
from torch.autograd import Variable

from src.peters_stuff.get_celeb_a import get_celeb_a_iterator
from src.peters_stuff.image_crop_generator import get_image_batch_crop_generator


# demo_dbplot()


# def train(epoch):
#     VAE_model.train()
#     GAN_model.train()
#
#     train_loss = [0, 0]
#     for batch_idx, data in enumerate(train_loader):
#         labels = torch.zeros(data.shape[0], 1)
#         labels = Variable(labels).double()
#         noise_variable = torch.zeros(data.shape[0], latent_dims).double().normal_()
#         noise_variable = Variable(noise_variable).double()
#         data = Variable(data)
#         if args.cuda:
#             data = data.cuda()
#             labels = labels.cuda()
#             noise_variable = noise_variable.cuda()
#
#
#         # Optimize discriminator
#         GAN_model.zero_grad()
#         labels.fill_(1)
#         #noise_variable.resize_(data.size(0), latent_dims, 1, 1)
#
#         predicted_real_labels  = GAN_model(data)
#         real_GAN_loss = GAN_model.loss_function(predicted_real_labels, labels)
#         real_GAN_loss.backward()
#
#         gen_data = VAE_model.decode(noise_variable)
#         labels.fill_(0)
#         predicted_fake_labels = GAN_model(gen_data.detach())
#         fake_GAN_loss = GAN_model.loss_function(predicted_fake_labels, labels)
#         fake_GAN_loss.backward()
#         GAN_opt.step()
#
#         GAN_loss = real_GAN_loss.data[0] + fake_GAN_loss.data[0]
#         train_loss[0] += GAN_loss
#
#         # Optimize VAE
#         VAE_model.zero_grad()
#         recon_batch, mu, logvar = VAE_model(data)
#
#         labels.fill_(1)
#         predicted_gen_labels = GAN_model.discriminate(recon_batch)
#         rec_loss, gen_loss = VAE_model.loss_function(recon_batch, data, mu,
#                                                      logvar, predicted_gen_labels, labels)
#         rec_loss.backward(retain_graph = True)
#         gen_loss.backward()
#         VAE_opt.step()
#
#         VAE_loss = rec_loss.data[0] + gen_loss.data[0]
#         train_loss[1] += VAE_loss
#
#         if batch_idx % args.log_interval == 0:
#             print('Train epoch: {} [{}/{} ({:.0f}%]\nGAN loss: {:.6f}, VAE loss: {:.6f}'.format(
#                     epoch,
#                     batch_idx * len(data),
#                     len(train_loader.dataset),
#                     100. * batch_idx / len(train_loader),
#                     GAN_loss / len(data),
#                     VAE_loss / len(data),
#                     ))
#
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#         epoch, (train_loss[0] + train_loss[1]) / len(train_loader.dataset)))


# def test(epoch):
#     VAE_model.eval()
#     GAN_model.eval()
#
#     # Don't test every epoch
#     if epoch % 5:
#         return
#
#     test_loss = 0
#     for i, data in enumerate(test_loader):
#         labels = torch.ones(data.shape[0], 1)
#         labels = Variable(labels).double()
#         data = Variable(data, volatile = True)
#         if args.cuda:
#             data = data.cuda()
#             labels = labels.cuda()
#
#         recon_batch, mu, logvar = VAE_model(data)
#         disc_recon_data = GAN_model(recon_batch)
#         loss = VAE_model.loss_function(recon_batch, data, mu, logvar, disc_recon_data, labels)
#         test_loss += loss[0].data[0] + loss[1].data[0]
#
#         if i == 0 and epoch % 50 == 0:
#             n = min(data.size(0), 8)
#             comparison = torch.cat([data[:n],
#                                    recon_batch.view(args.batch_size, *size)[:n]])
#             save_image(comparison.data.cpu(), 'results/reconstruction_with_GAN_' + str(epoch) + '.png', nrow = n)
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))
from src.peters_stuff.sweeps import generate_linear_sweeps


@experiment_function
def demo_train_vaegan_on_images(
        batch_size=64,
        cuda=False,
        seed=1234,
        checkpoints={0:10, 100:100, 1000: 1000},
        learning_rate=1e-3,
        latent_dims = 8,
        image_size = (64, 64),
        ):

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # size = (3, *image_size)
    VAE_model = VAE(latent_dims, image_size).float()
    GAN_model = Discriminator(latent_dims, image_size).float()
    if cuda:
        VAE_model.cuda()
        GAN_model.cuda()
    VAE_opt = optim.Adam(VAE_model.parameters(), lr = learning_rate, betas = (0.5, 0.999))
    GAN_opt = optim.Adam(GAN_model.parameters(), lr = learning_rate, betas = (0.5, 0.999))

    def train_step(data):
        labels = torch.zeros(data.shape[0], 1)
        labels = Variable(labels).float()
        noise_variable = torch.zeros(data.shape[0], latent_dims).float().normal_()
        noise_variable = Variable(noise_variable).float()
        data = Variable(data)
        if cuda:
            data = data.cuda()
            labels = labels.cuda()
            noise_variable = noise_variable.cuda()


        # Optimize discriminator
        GAN_model.zero_grad()
        labels.fill_(1)

        predicted_real_labels  = GAN_model(data)
        real_GAN_loss = GAN_model.loss_function(predicted_real_labels, labels)
        real_GAN_loss.backward()

        gen_data = VAE_model.decode(noise_variable)
        labels.fill_(0)
        predicted_fake_labels = GAN_model(gen_data.detach())
        fake_GAN_loss = GAN_model.loss_function(predicted_fake_labels, labels)
        fake_GAN_loss.backward()
        GAN_opt.step()

        GAN_loss = real_GAN_loss.data[0] + fake_GAN_loss.data[0]

        # Optimize VAE
        VAE_model.zero_grad()
        recon_batch, mu, logvar = VAE_model(data)

        labels.fill_(1)
        predicted_gen_labels = GAN_model.discriminate(recon_batch)
        rec_loss, gen_loss = VAE_model.loss_function(recon_batch, data, mu,
                                                     logvar, predicted_gen_labels, labels)
        rec_loss.backward(retain_graph = True)
        gen_loss.backward()
        VAE_opt.step()

        VAE_loss = rec_loss.data[0] + gen_loss.data[0]

        return VAE_loss, GAN_loss

    is_checkpoint = Checkpoints(checkpoints)

    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')

    cut_size = 128
    img = img[img.shape[0]//2-cut_size//2:img.shape[0]//2+cut_size//2, img.shape[1]//2-cut_size//2:img.shape[1]//2+cut_size//2]  # TODO: Revert... this is just to test on a smaller version

    dbplot(img, 'full_img')

    # for i, image_crops in enumerate(get_celeb_a_iterator(minibatch_size=batch_size, size=image_size)):
    mode = 'random'
    for i, (bboxes, image_crops) in enumerate(get_image_batch_crop_generator(img=img, crop_size=image_size, batch_size=batch_size, mode=mode, speed=10, randomness=0.1)):

        image_crops = (image_crops.astype(np.float32))/256

        var_image_crops = torch.Tensor(np.rollaxis(image_crops, 3, 1))
        if cuda:
            var_image_crops = var_image_crops.cuda()
        vae_loss, gan_loss = train_step(var_image_crops)

        rate = 1/measure_period('train_step')
        if do_every('5s'):
            print(f'Iter: {i}, Iter/s: {rate:.3g}, VAE-Loss: {vae_loss:.3g}, GAN-Loss: {gan_loss:.3g}')

        if is_checkpoint():
            print('Checkping')

            recons, _, _ = VAE_model(var_image_crops)

            z_points = torch.randn([batch_size, latent_dims]).float()
            # z_grid = torch.Tensor(np.array(np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))).reshape(2, -1).T)

            n_sweep_samples = 8
            n_sweep_points = 8
            z_grid = torch.Tensor(generate_linear_sweeps(starts = np.random.randn(n_sweep_samples, latent_dims), ends=np.random.randn(n_sweep_samples, latent_dims), n_points=n_sweep_points).reshape(n_sweep_points*n_sweep_samples, latent_dims))
            if cuda:
                z_points = z_points.cuda()
                z_grid = z_grid.cuda()

            samples = VAE_model.decode(z_points)
            grid_samples = VAE_model.decode(z_grid)


            with hold_dbplots():
                # dbplot(image_crops, 'crops')
                dbplot(np.rollaxis(recons.detach().cpu().numpy(), 1, 4), 'recons')
                dbplot(np.rollaxis(samples.detach().cpu().numpy(), 1, 4), 'samples', cornertext = f'Iter {i}')
                dbplot(np.rollaxis(grid_samples.detach().cpu().numpy().reshape((n_sweep_samples, n_sweep_points, 3)+image_size), 2, 5), 'sweeps', cornertext = f'Iter {i}')



    #
    # width = 2000D
    # crop_size = (200, 200)
    # batch_size = 16
    # mode='smooth'  # 'smooth': Drifts randomly around crop space.  'random' gives random crops
    # path = get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')
    # img = resize_image(smart_load_image(path), width=width, mode='preserve_aspect')
    # if mode=='smooth':
    #     speed = 10
    #     randomness = 0.1
    #     bbox_gen_func = lambda im_shape: generate_smoothly_varying_bboxes(img_size = img.shape[:2], crop_size=crop_size, speed=speed, jitter=randomness)
    # elif mode=='random':
    #     bbox_gen_func = lambda im_shape: generate_random_bboxes(img_size = img.shape[:2], crop_size=crop_size)
    # else:
    #     raise Exception()
    #
    # dbplot(img, 'image')
    # for bboxes, image_crops in get_image_batch_crop_generator(bbox_gen_func, img=img, crop_size=crop_size, batch_size=batch_size):


#
# for epoch in range(1, args.epochs + 1):
#     train(epoch)
#     test(epoch)
#     if epoch % 50 == 0:
#         sample = Variable(torch.randn(64, latent_dims)).double()
#         if args.cuda:
#             sample = sample.cuda()
#         sample = VAE_model.decode(sample).cpu()
#         save_image(sample.data.view(64, *size),
#                    'results/sample_VAE_GAN_' + str(epoch) + '.png')
#
#     # Save model
#     if args.save_path and not epoch % 50:
#         save_file = '{0}VAE_model_learning-rate_{1}_batch-size_{2}_epoch_{3}_nr-images_{4}.pt'.format(
#                     args.save_path,
#                     args.learning_rate,
#                     args.batch_size,
#                     epoch,
#                     args.nr_images)
#         torch.save(VAE_model, save_file)
#         save_file = '{0}GAN_model_learning-rate_{1}_batch-size_{2}_epoch_{3}_nr-images_{4}.pt'.format(
#                     args.save_path,
#                     args.learning_rate,
#                     args.batch_size,
#                     epoch,
#                     args.nr_images)
#         torch.save(GAN_model, save_file)


if __name__ == '__main__':
    demo_train_vaegan_on_images(cuda=True, latent_dims=20)