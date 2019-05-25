
from utils import read_tiff, write_tiff
import numpy as np
import os.path as osp
import argparse
from glob import glob


def labelPlanarInstances(planes):

    h, w, ch = planes.shape
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
    for filename in filenames:
        planes = read_tiff(filename)
        planes = labelPlanarInstances(planes)
        return


if __name__ == '__main__':
    main()
