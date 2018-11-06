from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.image_ops import resize_image


def get_sistine_cut(image_cut_size, rescale_width):
    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=rescale_width, mode='preserve_aspect')
    img = img[img.shape[0]//2-image_cut_size[0]//2:img.shape[0]//2+image_cut_size[0]//2, img.shape[1]//2-image_cut_size[1]//2:img.shape[1]//2+image_cut_size[1]//2]  # TODO: Revert... this is just to test on a smaller version
    return img


class SampleImages:

    @staticmethod
    def sistine_512():
        return get_sistine_cut(image_cut_size=(512, 512), rescale_width=2000)
