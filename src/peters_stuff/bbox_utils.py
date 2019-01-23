import numpy as np
from artemis.general.should_be_builtins import izip_equal

"""
A bunch of bounding box functions.

All bounding boxes are assumed to be in LTRB format unless otherwise stated.
"""


class BBoxFormats(object):
    LTWH1 = 'ltwh1'
    LTWH = 'ltwh'
    LTRB1 = 'ltrb1'
    LTRB = 'ltrb'
    XYWH = 'xywh'


_bbox_converters = {
    (BBoxFormats.LTWH1, BBoxFormats.LTRB): lambda lbwh: (lbwh[0] - 1., lbwh[1] - 1., lbwh[0] + lbwh[2] - 1., lbwh[1] + lbwh[3] - 1.),
    (BBoxFormats.LTWH1, BBoxFormats.LTRB1): lambda lbwh: (lbwh[0], lbwh[1], lbwh[0] + lbwh[2] - 1., lbwh[1] + lbwh[3] - 1.),
    (BBoxFormats.LTRB, BBoxFormats.XYWH): lambda ltrb: ((ltrb[0]+ltrb[2])/2., (ltrb[3]+ltrb[1])/2., float(ltrb[2]-ltrb[0]), float(ltrb[3]-ltrb[1])),
    (BBoxFormats.XYWH, BBoxFormats.LTRB): lambda xywh: (xywh[0]-xywh[2]/2., xywh[1]-xywh[3]/2., xywh[0]+xywh[2]/2., xywh[1]+xywh[3]/2.),

}


def convert_bbox(bbox, src_format, dest_format):
    """
    Convert between bounding box formats:

    lbrt: left, bottom, right, top: Exclusive.
    lbrt1: left, bottom, right, top, with indexing starting from 1

    :param bbox:
    :param src_format:
    :param dest_format:
    :return:
    """
    if src_format == dest_format:
        return bbox
    else:
        return _bbox_converters[src_format, dest_format](bbox)


def expand_bbox(bbox, factor):
    """
    :param bbox: An lbrt bbox
    :param factor: The factor by which you'd like to expand
    :return: An expanded lbrt bbox
    """
    l, b, r, t = bbox
    edge_pad = (factor - 1) / 2.
    w_pad = (r - l) * edge_pad
    h_pad = (t - b) * edge_pad
    return l - w_pad, b - h_pad, r + w_pad, t + h_pad


def scale_bbox(bbox, scale, src_format = BBoxFormats.LTRB):
    """
    Same as expand bbox, but we express it in scale, instead of factor of difference.
    :param bbox:
    :param scale:
    :return:
    """
    x, y, w, h = convert_bbox(bbox, src_format, BBoxFormats.XYWH)
    return convert_bbox((x, y, w*scale, h*scale), BBoxFormats.XYWH, src_format)


def full_frame_bbox(im_shape):
    """
    Return a bbox
    :param im_shape:
    :param previous_scaling:
    :return:
    """
    h, w = im_shape
    return 0., 0., float(w), float(h)


def bbox_str(bbox):
    return '['+', '.join('{:4.2f}'.format(x) for x in bbox) + ']'


def scaling_bbox(im_shape, scale):
    """
    Get a bbox that will create a crop of (scale) times the size of the original image.
    :param im_shape: The (y, x) shape of the image
    :param scale: A float
    :return: A new bbox that is scale times the size of the image.
    """
    return scale_bbox(full_frame_bbox(im_shape), scale=scale)


def compute_bbox_delta(bbox1, bbox2):
    x1, y1, w1, h1 = convert_bbox(bbox1, BBoxFormats.LTRB, BBoxFormats.XYWH)
    x2, y2, w2, h2 = convert_bbox(bbox2, BBoxFormats.LTRB, BBoxFormats.XYWH)
    # return (x2-x1)/w1, (y2-y1)/h1, (w2-w1)/w1, (h2-h1)/h1
    return (x2-x1)/w1, (y2-y1)/h1, np.log2(w2/w1), np.log2(h2/h1)


def remap_bbox(bbox, cutting_bbox):
    """
    :param bbox: Given a bbox, remap it so that it is in the reference frame of the image after it is cut by the cutting bbox.
    :param cutting_bbox:
    :return: A new bbox
    """
    lc, tc, _, _ = cutting_bbox
    l, t, r, b = bbox
    return l-lc, t-tc, r-lc, b-tc


def apply_bbox_delta(bbox, delta):
    """

    :param bbox:
    :param delta:
    :return:
    """
    assert is_bbox_valid(bbox), '{} is an invalid bbox'.format(bbox)
    x1, y1, w1, h1 = convert_bbox(bbox, BBoxFormats.LTRB, BBoxFormats.XYWH)
    dx, dy, dw, dh = delta
    # x2, y2, w2, h2 = dx*w1+x1, dy*h1+y1, dw*w1+w1, dh*h1+h1
    x2, y2, w2, h2 = dx*w1+x1, dy*h1+y1, w1*2**dw, h1*2**dh
    result = convert_bbox((x2, y2, w2, h2), BBoxFormats.XYWH, BBoxFormats.LTRB)

    assert is_bbox_valid(result), '{} is an invalid bbox'.format(result)
    return result


def apply_delta_to_scaled_crop(im_shape, delta, scale):
    """
    Given an image shape, taken from an image that was cut by a bbox and then scaled,
    and a delta bbox.  Infer the position of the next bbox.

        query_im = crop_img_with_bbox(im, scale_bbox(bbox, scale))
        new_bbox_relative_to_query = apply_delta_to_scaled_crop(im.shape[:2], delta, scale)

    :param im_shape: An (y, x) shape of a query image
    :param delta: The bbox delta to apply
    :param scale: The scale that was used when cropping the query image
    :return: new_bbox_relative_to_query: The position of the next bbox, relative to the query frame.
    """
    original_bbox_in_query_frame = scaling_bbox(im_shape, scale=1./scale)
    return apply_bbox_delta(original_bbox_in_query_frame, delta)


def apply_delta_to_full_frame(frame_shape, delta):
    bbox = full_frame_bbox(frame_shape)
    return apply_bbox_delta(bbox, delta)


def is_bbox_valid(bbox):
    l, t, r, b = bbox
    return l<=r and t<=b


def get_bbox_overlap(bbox1, bbox2, check_validity=True):
    """
    Get overlap (intersection over union) between bounding boxes.

    Thanks Martin: https://stackoverflow.com/a/42874377/851699

    :param bbox1: A LTRB bounding box
    :param bbox2: A LTRB bounding box
    :return: A number between 0 and 1 indicating the overlap.
    """

    if check_validity:
        assert is_bbox_valid(bbox1), 'Bounding Box 1 is invalid: {}'.format(bbox1)
        assert is_bbox_valid(bbox2), 'Bounding Box 2 is invalid: {}'.format(bbox1)
    l1, t1, r1, b1 = bbox1
    l2, t2, r2, b2 = bbox2
    area1 = (r1 - l1) * (b1 - t1)
    area2 = (r2 - l2) * (b2 - t2)
    li = max(l1, l2)
    ti = max(t1, t2)
    bi = min(b1, b2)
    ri = min(r1, r2)
    if ri < li or bi < ti:
        return 0.0
    else:
        intersection_area = (ri - li) * (bi - ti)
        iou = intersection_area / float(area1 + area2 - intersection_area)
        return iou


def compute_overlap_from_deltas(delta1, delta2, check_validity=True):

    fake_bbox = (0., 0., 1., 1.)

    bbox1 = apply_bbox_delta(fake_bbox, delta1)
    bbox2 = apply_bbox_delta(fake_bbox, delta2)

    return get_bbox_overlap(bbox1, bbox2, check_validity=check_validity)


def clip_to_range(val, minimum, maximum):
    return min(max(val, minimum), maximum)


def shift_bbox(bbox, shift):
    """
    :param bbox: An LTRB bbox
    :param shift: An (x, y) shift
    :return: A new, shifted, ltrb bbox.
    """
    l, t, r, b = bbox
    dx, dy = shift
    return l+dx, t+dy, r+dx, b+dy


def clip_bbox_to_image(bbox, width, height, preserve_size=False):
    """
    :param bbox: A ltrb 1-indexed bounding box.
    :param width: Image width
    :param height: Image height
    :return: A ltrb 1-indexed bounding box clipped to the image.
    """
    l, t, r, b = bbox
    if preserve_size:
        sx, sy = r-l, b-t
        lnew = clip_to_range(l, 1, width-sx)
        tnew = clip_to_range(t, 1, height-sy)
        return lnew, tnew, lnew+sx, tnew+sy
    else:
        return clip_to_range(l, 0, width), clip_to_range(t, 0, height), clip_to_range(r, 0, width), clip_to_range(b, 0, height)


def crop_img_with_bbox(img, bbox, crop_edge_setting = 'cut'):
    """

    :param img: A (size_y, size_x, 3) image
    :param bbox: A LTRB bbox
    :param crop_edge_setting:
        cut: Just cut off edges (image will be smaller when bbox hangs off edge)
        pad_rep: Zero-pad edges
    :return: A cropped image
    """
    if crop_edge_setting=='cut':
        l, t, r, b = np.clip(np.round(bbox).astype(np.int), [0, 0, 0, 0], [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        return img[int(t):int(b), int(l):int(r)]
    elif crop_edge_setting=='pad_rep':
        l, t, r, b = np.round(bbox).astype(int)
        y_ixs = np.clip(np.arange(t, b), 0, img.shape[0]-1)
        x_ixs = np.clip(np.arange(l, r), 0, img.shape[1]-1)
        return img[y_ixs[:, None], x_ixs]
    elif crop_edge_setting=='error':
        l, t, r, b = np.round(bbox).astype(np.int)
        assert l>=0 and t>=0 and r<=img.shape[1] and b<=img.shape[0], f'bbox {(l, t, r, b)} did not fit in image of size {img.shape}'
        return img[int(t):int(b), int(l):int(r)]
    else:
        raise Exception(crop_edge_setting)


def bound_bbox_delta(predicted_delta, true_delta, max_delta):
    """

    :param predicted_delta: A desired bbox delta (dx, dy, dw, dh)
    :param true_delta: The true delta (dx, dy, dw, dh)
    :param max_delta: The (max|dx|, max|dy|, max|dw|, max|dh|)
    :return:
    """
    return tuple(np.clip(p, t-m, t+m) for p, t, m in izip_equal(predicted_delta, true_delta, max_delta))


def bbox_to_position(bboxes, image_size):
    bboxes = np.array(bboxes)
    return bboxes[:, [1, 0]] / np.array((image_size[0] - (bboxes[:, 3]-bboxes[:, 1]), image_size[1] - (bboxes[:, 2]-bboxes[:, 0]))).T - 0.5


def position_to_bbox(positions, image_size, crop_size, clip=False):

    if clip:
        positions = np.clip(positions, -.5, .5)

    unnorm_positions = ((np.array(positions)+.5)[:, [1, 0]] * (image_size[0] - crop_size[0], image_size[1] - crop_size[1])).astype(np.int)
    bboxes = np.concatenate([unnorm_positions, unnorm_positions+crop_size], axis=1)
    return bboxes
