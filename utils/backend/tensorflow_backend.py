import tensorflow


def disable_tensorflow_v2_behavior():
    """ See https://www.tensorflow.org/api_docs/python/tf/compat/v1/disable_tensorflow_v2_behavior .
    """
    tensorflow.compat.v1.disable_v2_behavior()


def ones(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/ones .
    """
    return tensorflow.ones(*args, **kwargs)


def transpose(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/transpose .
    """
    return tensorflow.transpose(*args, **kwargs)


def map_fn(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/map_fn .
    """
    return tensorflow.map_fn(*args, **kwargs)


def pad(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/pad .
    """
    return tensorflow.pad(*args, **kwargs)


def top_k(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/nn/top_k .
    """
    return tensorflow.nn.top_k(*args, **kwargs)


def clip_by_value(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/clip_by_value .
    """
    return tensorflow.clip_by_value(*args, **kwargs)


def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/image/resize_images .

    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        'bilinear': tensorflow.image.ResizeMethod.BILINEAR,
        'nearest' : tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic' : tensorflow.image.ResizeMethod.BICUBIC,
        'area'    : tensorflow.image.ResizeMethod.AREA,
    }
    return tensorflow.compat.v1.image.resize_images(images, size, tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR, False)


def non_max_suppression(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression .
    """
    return tensorflow.image.non_max_suppression(*args, **kwargs)


def range(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/range .
    """
    return tensorflow.range(*args, **kwargs)


def scatter_nd(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/scatter_nd .
    """
    return tensorflow.scatter_nd(*args, **kwargs)


def gather_nd(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/gather_nd .
    """
    return tensorflow.gather_nd(*args, **kwargs)


def meshgrid(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/meshgrid .
    """
    return tensorflow.meshgrid(*args, **kwargs)


def where(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/where .
    """
    return tensorflow.where(*args, **kwargs)


def unstack(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/unstack .
    """
    return tensorflow.unstack(*args, **kwargs)
