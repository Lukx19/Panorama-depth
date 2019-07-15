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
    config_file, result_dir = args
    depth_folder, filename = osp.split(config_file)
    scan_name = filename[:-5]
    result_scan_folder = osp.join(result_dir, scan_name)
    os.makedirs(result_scan_folder, exist_ok=True)

    cmd = f"./matterport_panos/bin/convert2pano '{config_file}' '{result_scan_folder}'"
    os.popen(cmd).read()


def main():
    parser = argparse.ArgumentParser(
        description='Covert Stanford dataset to Omidepth dataset format')
    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument('result_dir', type=str,
                        help='All generated files will be saved here')
    args = parser.parse_args()

    if (not osp.exists(args.dataset)):
        print("Directory: {} does not exist. Skipping.".format(args.dataset))
        return
    os.makedirs(args.result_dir, exist_ok=True)
    config_files = glob.glob(args.dataset + '/**/*.conf', recursive=True)
    print(config_files)
    input_args = [(file, args.result_dir) for file in config_files]
    # pool = Pool(2 * os.cpu_count() // 3)
    pool = Pool(1)
    for _ in tqdm.tqdm(pool.imap_unordered(process, input_args), total=len(input_args)):
        pass


if __name__ == '__main__':
    main()
