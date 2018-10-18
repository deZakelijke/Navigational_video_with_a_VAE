from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VideoData import VideoData
from VAE_with_Disc import VAE, Discriminator
from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints
from artemis.general.image_ops import resize_image
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from src.peters_stuff.image_crop_generator import get_image_batch_crop_generator


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





def demo_train_vaegan(
        batch_size=32,
        epochs=10,
        cuda=False,
        seed=1234,
        checkpoints={0:10, 100:100, 1000: 1000},
        learning_rate=1e-3,
        ):

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    latent_dims = 8
    image_size = (64, 64)
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

    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')

    is_checkpoint = Checkpoints(checkpoints)
    mode = 'random'
    for i, (bboxes, image_crops) in enumerate(get_image_batch_crop_generator(img=img, crop_size=image_size, batch_size=batch_size, mode=mode, speed=10, randomness=0.1)):

        image_crops = (image_crops.astype(np.float32))/256

        var_image_crops = torch.Tensor(np.rollaxis(image_crops, 3, 1))
        train_step(var_image_crops)

        if is_checkpoint():

            recons, _, _ = VAE_model(var_image_crops)

            z_points = torch.randn([batch_size, latent_dims]).float()
            samples = VAE_model.decode(z_points)

            with hold_dbplots():
                dbplot(image_crops, 'crops')
                dbplot(np.rollaxis(recons.detach().numpy(), 1, 4), 'recons')
                dbplot(np.rollaxis(samples.detach().numpy(), 1, 4), 'samples', cornertext = f'Iter {i}')



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
    demo_train_vaegan()