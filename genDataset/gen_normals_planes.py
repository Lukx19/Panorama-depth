import argparse
from multiprocessing import Pool
import glob
import os
import os.path as osp
import tqdm


def process_depth(fname):
    cmd = "./plane_normal_extraction/bin/segmentation '{}'".format(fname)
    os.popen(cmd).read()


def main():
    parser = argparse.ArgumentParser(
        description="Adds normals and smooth plannary surface segmentation \
                     to dataset with *.tiff depth files")
    parser.add_argument('dataset', type=str, help='Path to dataset')
    args = parser.parse_args()

    if (not osp.exists(args.dataset)):
        print("Directory: {} does not exist. Skipping.".format(args.dataset))
        return
    filenames = glob.glob(args.dataset + '/**/*_depth_*.tiff', recursive=True)

    pool = Pool(2 * os.cpu_count() // 3)
    for _ in tqdm.tqdm(pool.imap_unordered(process_depth, filenames), total=len(filenames)):
        pass


if __name__ == '__main__':
    main()
