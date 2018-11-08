from typing import Tuple, Iterator
import numpy as np
import itertools
from artemis.fileman.file_getter import get_file
from artemis.general.numpy_helpers import get_rng
from artemis.ml.tools.iteration import batchify_generator
from src.peters_stuff.bbox_utils import crop_img_with_bbox


POSITION_GENERATOR = Iterator[Tuple[int, int]]


def iter_pos_drift(n_dim, speed, cell_size=1., jitter=0.1, rng=None) -> POSITION_GENERATOR:
    """
    Simulate an object moving around in a cell at a given speed, bouncing off the walls whenever it hits one.
    :param int n_dim:
    :param float speed:
    :param Union[Sequence[float],float] cell_size:
    :param rng:
    :return Generator[Tuple[float, float]]:
    """
    rng = get_rng(rng)
    v_raw = rng.randn(2)
    p = rng.rand(n_dim)*cell_size

    cell_size = np.array([cell_size]*n_dim) if np.isscalar(cell_size) else np.array(cell_size)
    assert 0 <= jitter <= 1
    assert np.all(speed < np.array(cell_size)), "Things get weird if speed is greater than cell size"
    for _ in itertools.count(0):
        yield p
        p = p + speed*v_raw/np.sqrt((v_raw**2).sum())
        has_hit = (p<0) | (p>cell_size)
        v_raw[has_hit] = -v_raw[has_hit]
        p[has_hit] = np.mod(-p[has_hit], cell_size[has_hit])
        v_raw = (1-jitter)*v_raw + jitter*rng.randn(2)


def iter_pos_random(n_dim, rng) -> POSITION_GENERATOR:
    rng = get_rng(rng)
    yield from (rng.rand(n_dim) for _ in itertools.count(0))


def iter_pos_expanding(position_generator: POSITION_GENERATOR, n_iters, center=(0.5, 0.5), start_range = (0, 0), end_range=(1, 1)) -> POSITION_GENERATOR:
    """
    Imagine a box centered at (0.5, 0.5) that grows from start_range to end_range over n_iter iterations.
    Positions yielded by the inner generator are placed within this box
    :param position_generator: Yields positions in range (0, 1)
    :param n_iters: Number of iterations oves which to expand from start range to end range
    :param center: Center-point from which you expand.
    :param start_range:
    :param end_range:
    :return:
    """
    start_range = np.array(start_range)
    end_range = np.array(end_range)
    center = np.array(center)

    for t, rel_position in enumerate(position_generator):
        frac = min(1, t/n_iters)
        current_range = (1-frac)*start_range + frac * end_range
        position = (rel_position-center)*current_range + center
        yield position


def generate_relative_position_crop(img_size, crop_size, crop_position):
    size_y, size_x = img_size
    crop_size_y, crop_size_x = crop_size
    crop_pos_y, crop_pos_x = crop_position
    l = int((size_x-crop_size_x)*crop_pos_x)
    t = int((size_y-crop_size_y)*crop_pos_y)
    return l, t, l+crop_size_x, t+crop_size_y


def generate_random_bboxes(img_size, crop_size, rng=None):
    """
    Generate LTRB bounding boxes.
    :param img_size: (size_y, size_x) of images
    :param crop_size: (size_y, size_x) of crop
    :param rng: Optionally, a random number generator
    :return Tuple[int]: A (left, right, top, bottom) bounding box
    """
    yield from (generate_relative_position_crop(img_size=img_size, crop_size=crop_size, crop_position=position) for position in iter_pos_random(rng))


def generate_smoothly_varying_bboxes(img_size, crop_size, speed, jitter=0.1, rng=None):
    yield from (generate_relative_position_crop(img_size=img_size, crop_size=crop_size, crop_position=position) for position in iter_pos_drift(n_dim=2, speed=speed, cell_size=1., jitter=jitter, rng=rng))


def iter_bboxes_from_positions(img_size, crop_size, position_generator):
    """
    :param img_size:
    :param crop_size:
    :param position_generator: A generator yielding (y_pos, x_pos) where each of y_pos, x_pos is in [0, 1]
        e.g. rel_position_generator = (np.random.rand(2) for _ in itertools.count(0))
    """
    for rel_position in position_generator:
        yield generate_relative_position_crop(img_size=img_size, crop_size=crop_size, crop_position=rel_position)


def iter_bbox_batches(image_shape, crop_size, batch_size, position_generator_constructor ='random', rng=None):

    rng = get_rng(rng)
    if isinstance(position_generator_constructor, str):
        if position_generator_constructor== 'random':
            position_generator_constructor = lambda: iter_pos_random(n_dim=2, rng=rng)
        else:
            raise NotImplementedError(position_generator_constructor)
    else:
        position_generator_constructor = position_generator_constructor

    batched_bbox_generator = batchify_generator(list(
            iter_bboxes_from_positions(
                img_size=image_shape,
                crop_size=crop_size,
                position_generator=position_generator_constructor(),
            ) for _ in range(batch_size)))

    yield from batched_bbox_generator


def batch_crop(img, bboxes):

    first = crop_img_with_bbox(img, bbox = bboxes[0], crop_edge_setting='error')
    batch = np.zeros((len(bboxes), *first.shape), dtype=first.dtype)
    batch[0] = first
    for i, bbox in enumerate(bboxes[1:]):
        batch[i+1] = crop_img_with_bbox(img, bbox = bbox, crop_edge_setting='error')
    return batch


if __name__ == "__main__":
    from artemis.fileman.smart_io import smart_load_image
    from artemis.general.image_ops import resize_image
    from artemis.plotting.db_plotting import dbplot, DBPlotTypes, hold_dbplots

    # Here we demonstrate our data-generating process for the "crop from image" experiments, wherein we generate a bunch
    # of crops from a given image.
    CROP_SIZE = (200, 200)
    BATCH_SIZE = 16
    IMG = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')

    # crop_gen_func = lambda: generate_random_bboxes(img_size=img.shape[:2], crop_size=crop_size)
    # crop_gen_func = lambda: generate_expanding_random_bbox_range(img_size=img.shape[:2], crop_size=crop_size, rel_position_generator = (np.random.rand(2) for _ in itertools.count(0)), n_iters=100, start_range=(0.1, 0.1))
    crop_gen_func = lambda: iter_bboxes_from_positions(
        img_size=IMG.shape[:2],
        crop_size=CROP_SIZE,
        position_generator= iter_pos_expanding(
            position_generator=iter_pos_drift(n_dim=2, speed=0.02, cell_size=1., jitter=0.1, rng=None), n_iters=100, start_range=(0.1, 0.1),
        ),
    )

    dbplot(IMG, 'image')
    for bboxes in batchify_generator((crop_gen_func() for _ in range(BATCH_SIZE)), batch_size=BATCH_SIZE):
        image_crops = batch_crop(img=IMG, bboxes=bboxes)
        with hold_dbplots():
            dbplot(image_crops, 'crops')
            for i, bbox in enumerate(bboxes):
                dbplot(bbox, f'bbox[{i}]', axis='image', plot_type=DBPlotTypes.BBOX_R)
