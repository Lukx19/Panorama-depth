import glob
import argparse
import OpenEXR
import numpy as np
import os.path as osp
import Imath
import array
from utils import write_tiff
from tqdm import tqdm


def read_exr(image_fpath):
    f = OpenEXR.InputFile(image_fpath)
    dw = f.header()['dataWindow']
    w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read in the EXR
    n_channels = len(f.header()["channels"])
    im = np.empty((h, w, n_channels))
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    if n_channels == 3:
        channels = f.channels(["R", "G", "B"], FLOAT)
    else:
        channels = f.channels(["Y"], FLOAT)

    for i, channel in enumerate(channels):
        im[:, :, i] = np.reshape(array.array('f', channel), (h, w))
    return im


def main():
    parser = argparse.ArgumentParser(
        description='Generate dataset with sparse depth images when provided with depth and rgb images')
    parser.add_argument('dataset', type=str, help='Path to dataset')
    args = parser.parse_args()

    if (not osp.exists(args.dataset)):
        print("Directory: {} does not exist.".format(args.dataset))
        return

    image_files = sorted(glob.glob(args.dataset + '/**/*.exr', recursive=True))
    for image_file in tqdm(image_files):
        depth = read_exr(image_file)[..., 0].astype(np.float32)
        write_tiff(image_file[:-3] + "tiff", depth)


if __name__ == '__main__':
    main()
