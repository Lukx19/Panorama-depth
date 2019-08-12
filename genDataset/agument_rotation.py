import argparse
from multiprocessing import Pool
import glob
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import utils
from PIL import Image
from scipy.spatial.transform import Rotation as R


def rotate(depth, color, angle, normals=None):
    width = depth.shape[1]
    pixels = int((width / 360) * angle)
    depth = np.roll(depth, pixels, axis=1)
    color = np.roll(color, pixels, axis=1)
    if normals is not None:
        h, w, ch = normals.shape
        print(normals.shape)
        normals = np.roll(normals, pixels, axis=1)
        n = normals[:, :, 0:3].reshape((-1, 3))
        # print(np.max(n))
        n = (n.astype(float) - 127.5) / 127.5
        rot = R.from_euler('z', angle, degrees=True)
        print(rot.apply(np.array([[0, 0, 0], [1, 0, 0]])))
        corrected_n = rot.apply(n)
        # corrected_n = n
        corrected_n = corrected_n.reshape((h, w, 3))
        corrected_n = np.trunc(corrected_n * 127.5 + 127.5).astype(np.uint8)
        # corrected_n = normals
        return depth, color, corrected_n
    return depth, color, None


def augmentWithRotation(files):
    """
        Adds 4 rotations to original depth and color files [0,90,180,275]
    Parameters
    ----------
    files :list of tuples (depth_file,color_file)
    """
    for file_types in tqdm(files):
        normal_file = None
        if len(file_types) == 2:
            depth_file, color_file = file_types
        else:
            depth_file, color_file, normal_file = file_types
        color = np.array(Image.open(color_file))
        depth = utils.read_tiff(depth_file)

        normal = None
        if normal_file:
            print(normal_file)
            normal = np.array(Image.open(normal_file))

        for angle in [90, 180, 275]:
            rot_depth_f = depth_file.replace("_0.", f"_{angle}.")
            rot_color_f = color_file.replace("_0.", f"_{angle}.")
            rot_depth, rot_color, rot_normal = rotate(depth, color, angle, normal)
            Image.fromarray(rot_color).save(rot_color_f)
            utils.write_tiff(rot_depth_f, rot_depth)
            if rot_normal is not None:
                rot_normal_f = normal_file.replace("_0.", f"_{angle}.")
                print(rot_normal.shape)
                Image.fromarray(rot_normal).save(rot_normal_f)


def main():
    parser = argparse.ArgumentParser(
        description="Adds 4 rotations to original depth and color files [-180,-90,0,90]")
    parser.add_argument('dataset', type=str, help='Path to dataset')
    args = parser.parse_args()

    if (not osp.exists(args.dataset)):
        print("Directory: {} does not exist. Skipping.".format(args.dataset))
        return
    depth_files = sorted(glob.glob(args.dataset + '/**/*_depth_0.tiff', recursive=True))
    color_files = sorted(glob.glob(args.dataset + '/**/*_color_0.png', recursive=True))
    # normal_files = sorted(glob.glob(args.dataset + '/**/*_normals_0.png', recursive=True))
    normal_files = []
    if len(normal_files) == len(depth_files):
        data = zip(depth_files, color_files, normal_files)
    else:
        data = zip(depth_files, color_files)
    augmentWithRotation(data)


if __name__ == '__main__':
    main()
