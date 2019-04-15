from skimage import io


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
