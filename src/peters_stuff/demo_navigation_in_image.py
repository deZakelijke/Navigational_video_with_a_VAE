import numpy as np


class INavigationModel(object):

    def plan_route(self, start_img, dest_img):
        raise NotImplementedError()


class PretrainedStrightLineNavModel(INavigationModel):

    def __init__(self, encoder, decoder, n_waypoints):

        self.encoder = encoder
        self.decoder = decoder
        self.frac = np.linspace(0, 1, n_waypoints)[:, None]

    def plan_route(self, start_img, dest_img):

        zs = self.encoder(start_img[None])
        zd = self.encoder(dest_img[None])
        zp = zs*(1-self.frac) + zd*self.frac
        video = self.decoder(zp)
        return video



def demo_vae_image_nav(model: INavigationModel):







