import numpy as np
from future.moves import itertools

from artemis.fileman.file_getter import get_file
from artemis.general.numpy_helpers import get_rng
from artemis.ml.tools.iteration import batchify_generator
from src.peters_stuff.bbox_utils import crop_img_with_bbox


def generate_bounded_random_drift(n_dim, speed, cell_size=1., jitter=0.1, rng=None):
    """
    Simulate an object moving around in a cell at a given speed, bouncing off the walls whenever it hits one.
    :param int n_dim:
    :param float speed:
    :param Union[Sequence[float],float] cell_size:
    :param rng:
    :return Sequence[float]:
    """
    rng = get_rng(rng)
    v_raw = rng.randn(2)
    p = rng.rand(n_dim)*cell_size

    cell_size = np.array(cell_size)
    assert 0 <= jitter <= 1
    assert np.all(speed < np.array(cell_size)), "Things get weird if speed is greater than cell size"
    for _ in itertools.count(0):
        yield p
        p = p + normalize(v_raw)*speed
        has_hit = (p<0) | (p>cell_size)
        v_raw[has_hit] = -v_raw[has_hit]
        p[has_hit] = np.mod(-p[has_hit], cell_size[has_hit])
        v_raw = (1-jitter)*v_raw + jitter*rng.randn(2)


def generate_random_bboxes(img_size, crop_size, rng=None):
    """
    Generate LTRB bounding boxes.
    :param img_size: (size_y, size_x) of images
    :param crop_size: (size_y, size_x) of crop
    :param rng: Optionally, a random number generator
    :return Tuple[int]: A (left, right, top, bottom) bounding box
    """
    rng = get_rng(rng)
    size_y, size_x = img_size
    crop_size_y, crop_size_x = crop_size
    for _ in itertools.count(0):
        l = rng.choice(size_x-crop_size_x)
        t = rng.choice(size_y-crop_size_y)
        yield l, t, l+crop_size_x, t+crop_size_y


def normalize(x):
    return x/np.sqrt((x**2).sum())


def generate_smoothly_varying_bboxes(img_size, crop_size, speed, jitter=0.1, rng=None):
    rng = get_rng(rng)
    crop_size_y, crop_size_x = crop_size
    for t, l in generate_bounded_random_drift(n_dim=2, speed=speed, cell_size=(img_size[0]-crop_size_y, img_size[1]-crop_size_x), jitter=jitter, rng=rng):
        l, t = int(l), int(t)
        yield l, t, l+crop_size_x, t+crop_size_y


def get_image_batch_crop_generator(img, crop_size, batch_size, mode, speed=10, randomness=0.1):
    """
    Convenience function for generating data

    :param Callable[[int,int], Generator[Tuple[int,int,int,int]] bbox_gen_func: A function that, given a (y,x) image size,
        ruturns a bounding-box generator.
    :param img: A (size_y, size_x, 3) image.
    :param crop_size: A crop size
    :param batch_size: The number of crops to batch together.
    :return Generator[Tuple[array[batch_size, 4], array[batch_size,crop_size[0],crop_size[1],3]]]: A generator which yields
        typles of bounding boxes and the images cropped by them.
    """
    im_shape = img.shape[:2]
    if mode=='smooth':
        speed = 10
        randomness = 0.1
        bbox_gen_func = lambda im_shape: generate_smoothly_varying_bboxes(img_size = img.shape[:2], crop_size=crop_size, speed=speed, jitter=randomness)
    elif mode=='random':
        bbox_gen_func = lambda im_shape: generate_random_bboxes(img_size = img.shape[:2], crop_size=crop_size)
    else:
        raise Exception()

    for bboxes in batchify_generator((bbox_gen_func(im_shape) for _ in itertools.count(0)), batch_size=batch_size):
        cropped_images = np.array([crop_img_with_bbox(img, bbox = bbox, crop_edge_setting='error') for bbox in bboxes])
        yield bboxes, cropped_images


if __name__ == "__main__":
    from artemis.fileman.smart_io import smart_load_image
    from artemis.general.image_ops import resize_image
    from artemis.plotting.db_plotting import dbplot, DBPlotTypes, hold_dbplots

    # Here we demonstrate our data-generating process for the "crop from image" experiments, wherein we generate a bunch
    # of crops from a given image.
    mode='smooth'  # 'smooth': Drifts randomly around crop space.  'random' gives random crops
    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')
    dbplot(img, 'image')
    for bboxes, image_crops in get_image_batch_crop_generator(img=img, crop_size=(200, 200), batch_size=16, mode=mode, speed=10, randomness=0.1):
        with hold_dbplots():
            dbplot(image_crops, 'crops')
            for i, bbox in enumerate(bboxes):
                dbplot(bbox, f'bbox[{i}]', axis='image', plot_type=DBPlotTypes.BBOX_R)
