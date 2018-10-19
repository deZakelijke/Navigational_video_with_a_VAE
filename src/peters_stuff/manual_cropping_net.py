from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.image_ops import resize_image


def manual_cropping_net():

    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')


