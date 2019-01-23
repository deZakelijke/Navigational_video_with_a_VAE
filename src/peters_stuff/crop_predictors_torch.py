from src.VAE_with_Disc import VAE
from src.peters_stuff.crop_predictors import ICropPredictor, imbatch_to_feat, feat_to_imbatch


class DeconvCropPredictor(ICropPredictor):

    def __init__(self, image_size, learning_rate=1e-3, filters=32, loss_type='mse'):
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = VAE(latent_dims=2, image_size=image_size, filters=filters).to(self.device)
        self.opt = torch.optim.Adam(list(self.model.parameters()), lr = learning_rate, betas = (0.5, 0.999))
        self.loss_type = loss_type

    def train(self, positions, image_crops):
        import torch
        predicted_imgs = self.model.decode(torch.Tensor(positions).to(self.device))
        var_image_crops = torch.Tensor(imbatch_to_feat(image_crops, channel_first=True, datarange=(0, 1))).to(self.device)

        # with hold_dbplots(draw_every=10):  # Just check that data normalization is right
        #     dbplot(predicted_imgs, 'preed', plot_type=DBPlotTypes.CIMG)
        #     dbplot(var_image_crops, 'crooops', plot_type=DBPlotTypes.CIMG)
        if self.loss_type=='bce':
            loss = torch.nn.functional.binary_cross_entropy(predicted_imgs, var_image_crops, size_average = False)
        elif self.loss_type=='mse':
            loss = torch.nn.functional.mse_loss(predicted_imgs, var_image_crops, size_average = True)
        else:
            raise NotImplementedError(self.loss_type)
        loss.backward()
        self.opt.step()
        return feat_to_imbatch(predicted_imgs.detach().cpu().numpy(), channel_first=True, datarange=(0, 1)), loss.detach().cpu().numpy()

    def predict(self, positions):
        import torch
        predicted_imgs = self.model.decode(torch.Tensor(positions).to(self.device))
        return feat_to_imbatch(predicted_imgs.detach().cpu().numpy(), channel_first=True, datarange=(0, 1))

    @staticmethod
    def get_constructor(learning_rate=1e-3, filters=32):
        return lambda batch_size, image_size: DeconvCropPredictor(image_size=image_size, learning_rate=learning_rate, filters=filters)