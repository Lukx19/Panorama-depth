from im2pano import combineViews
import utils
from PIL import Image
import os.path as osp
import numpy as np
import glob
import argparse
import os
from multiprocessing import Pool
import tqdm


def process(args):
    rgb_file, result_dir = args
    depth_file = rgb_file.replace("_rgb.jpg", "_depth.png")
    # normal_file = rgb_file.replace("_rgb.jpg", "_normal.png")

    pano_folder, filename = osp.split(rgb_file)
    scan_folder, _ = osp.split(pano_folder)
    pano_hash = filename[:-8]
    _, scan_name = osp.split(scan_folder)
    result_scan_folder = osp.join(result_dir, scan_name)
    os.makedirs(result_scan_folder, exist_ok=True)

    res_depth_file = osp.join(result_scan_folder, pano_hash + "_depth_0.tiff")
    res_color_file = osp.join(result_scan_folder, pano_hash + "_color_0.png")
    # res_normal_file = osp.join(result_scan_folder, pano_hash + "_normals_0.png")

    rgb = Image.open(rgb_file).transpose(Image.ROTATE_180)
    depth = Image.open(depth_file).transpose(Image.ROTATE_180)
    # normal = Image.open(normal_file).transpose(Image.ROTATE_180)

    final_size = (512, 256)
    rgb = rgb.resize(final_size, Image.LANCZOS)
    rgb.save(res_color_file)

    # normal = normal.resize(final_size, Image.NEAREST)
    # normal.save(res_normal_file)

    depth = depth.resize(final_size, Image.LINEAR)
    depth = np.array(depth).astype(float)
    depth = depth / 1000.0
    # print(np.max(depth), np.median(depth))
    utils.write_tiff(res_depth_file, depth)


def main():
    parser = argparse.ArgumentParser(
        description='Covert rendered Matterport dataset to Omidepth dataset format')
    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument('result_dir', type=str,
                        help='All generated files will be saved here')
    args = parser.parse_args()

    if (not osp.exists(args.dataset)):
        print("Directory: {} does not exist. Skipping.".format(args.dataset))
        return
    os.makedirs(args.result_dir, exist_ok=True)
    rgb_files = glob.glob(args.dataset + '/**/*rgb.jpg', recursive=True)
    print(rgb_files)
    input_args = [(file, args.result_dir) for file in rgb_files]
    pool = Pool(2 * os.cpu_count() // 3)
    # pool = Pool(1)
    for _ in tqdm.tqdm(pool.imap_unordered(process, input_args), total=len(input_args)):
        pass


if __name__ == '__main__':
    main()
