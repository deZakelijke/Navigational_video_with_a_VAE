import itertools

from artemis.fileman.disk_memoize import memoize_to_disk
from artemis.fileman.local_dir import get_artemis_data_path
import os
import numpy as np

from artemis.fileman.smart_io import smart_load, smart_load_image
from artemis.general.image_ops import resize_image
from artemis.general.numpy_helpers import get_rng


@memoize_to_disk
def _list_files(loc):
    files = os.listdir(loc)
    return list(os.path.join(loc, f) for f in files if f.endswith('.jpg'))


def get_celeb_a_iterator(minibatch_size, size=None, sizing_mode = 'squeeze', rng=None):

    for files in get_celeb_a_file_iterator(minibatch_size=minibatch_size, rng=rng):
        img_batch = []
        for f in files:
            img = smart_load_image(f)
            if size is not None:
                img = resize_image(img, width=size[0], height=size[1], mode=sizing_mode)
            img_batch.append(img)
        yield np.array(img_batch)

    # yield from (np.array([smart_load_image(f, max_resolution=size) for f in files]) for files in get_celeb_a_file_iterator(minibatch_size=minibatch_size, rng=rng))


def get_celeb_a_file_iterator(minibatch_size, rng=None):
    rng = get_rng(rng)
    files = _list_files(get_artemis_data_path('data/celeba/img_align_celeba'))
    ixs = rng.permutation(len(files))
    for i in range(0, len(files), minibatch_size):
        these_ixs = ixs[i:i+minibatch_size]
        yield [files[i] for i in these_ixs]
        # batch = np.array([smart_load(files[i]) for i in these_ixs])
        # yield batch


if __name__ == "__main__":
    from artemis.plotting.db_plotting import dbplot
    for x in get_celeb_a_iterator(minibatch_size=9, size=(64, 64)):
        dbplot(x, 'faces')
