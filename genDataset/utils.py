from skimage import io
import numpy as np


def read_tiff(image_fpath):
    img = io.imread(image_fpath)
    return img


def write_tiff(image_fpath, data):
    if len(data.shape) > 2 and data.shape[2] > 3:
        raise Exception("write tiff can write only data with up to 3 channels")

    if image_fpath[-5:] != ".tiff":
        io.imsave(image_fpath + ".tiff", data, check_contrast=False, compress=6)
    else:
        io.imsave(image_fpath, data, check_contrast=False, compress=6)


def onehottify(x, n=None):
    """1-hot encode x with the max value n (computed from data if n is None)."""
    mat = np.copy(np.asarray(x))
    dtype = mat.dtype
    shape = mat.shape
    if len(shape) > 1:
        mat = np.reshape(mat, (-1, 1))
    n = int(np.max(x) + 1) if n is None else n
    onehot = np.eye(n, dtype=dtype)[mat.astype(int)]
    if len(shape) > 1:
        onehot = np.reshape(onehot, (*shape, n))
        onehot = np.squeeze(onehot)
    return onehot
