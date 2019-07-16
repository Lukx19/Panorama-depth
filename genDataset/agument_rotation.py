import argparse
from multiprocessing import Pool
import glob
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import utils
from PIL import Image


def rotate(depth, color, angle):
    width = depth.shape[1]
    pixels = int((width / 360) * angle)
    depth = np.roll(depth, pixels, axis=1)
    color = np.roll(color, pixels, axis=1)
    return depth, color


def augmentWithRotation(files):
    """
        Adds 4 rotations to original depth and color files [0,90,180,275]
    Parameters
    ----------
    files :list of tuples (depth_file,color_file)
    """
    for depth_file, color_file in tqdm(files):
        color = np.array(Image.open(color_file))
        depth = utils.read_tiff(depth_file)
        for angle in [90, 180, 275]:
            rot_depth_f = depth_file.replace("_0.", f"_{angle}.")
            rot_color_f = color_file.replace("_0.", f"_{angle}.")
            rot_depth, rot_color = rotate(depth, color, angle)
            Image.fromarray(rot_color).save(rot_color_f)
            utils.write_tiff(rot_depth_f, rot_depth)


def main():
    parser = argparse.ArgumentParser(
        description="Adds 4 rotations to original depth and color files [-180,-90,0,90]")
    parser.add_argument('dataset', type=str, help='Path to dataset')
    args = parser.parse_args()

    if (not osp.exists(args.dataset)):
        print("Directory: {} does not exist. Skipping.".format(args.dataset))
        return
    depth_files = sorted(glob.glob(args.dataset + '/**/*_depth_*.tiff', recursive=True))
    color_files = sorted(glob.glob(args.dataset + '/**/*_color_*.png', recursive=True))
    augmentWithRotation(zip(depth_files, color_files))


if __name__ == '__main__':
    main()
