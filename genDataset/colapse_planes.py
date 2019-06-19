
from utils import read_tiff, write_tiff
import numpy as np
import os.path as osp
import argparse
from glob import glob
from tqdm import tqdm


def labelPlanarInstances(planes):
    if len(planes.shape) < 3:
        print("too few channels", planes.shape)
        return None
    h, w, ch = planes.shape
    if ch == 1:
        print("Only one channel -> only one plane in file")
        return None

    bin_planes = np.sign(planes)
    labeled_planes = bin_planes * np.reshape(np.arange(1, ch + 1), (1, 1, -1))
    colapsed_planes = np.sum(labeled_planes, axis=2, keepdims=True)
    colapsed_planes[colapsed_planes == 0] = 1
    colapsed_planes = colapsed_planes - 1
    return colapsed_planes


def main():

    parser = argparse.ArgumentParser(
        description='Colapses planes in multiple channels to single channel')
    parser.add_argument('dataset', type=str, help='Path to dataset')

    args = parser.parse_args()

    if (not osp.exists(args.dataset)):
        print("Directory: {} does not exist. Skipping.".format(args.dataset))
        return
    filenames = sorted(glob(args.dataset + '/**/*_planes_*.tiff', recursive=True))

    for filename in tqdm(filenames):
        planes = read_tiff(filename)
        planes = labelPlanarInstances(planes)
        if planes is not None:
            write_tiff(filename, planes)
        else:
            print("Something wrong: ", filename)


if __name__ == '__main__':
    main()
