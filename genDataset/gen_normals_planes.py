import argparse
from multiprocessing import Pool
import glob
import os
import os.path as osp
import tqdm
from utils import read_tiff


calc_missing = False


def processDepth(fname):
    if calc_missing:
        fname_normals = fname.replace("_depth_", "_normals_")
        fname_planes = fname.replace("_depth_", "_planes_")
        print(fname)
        if osp.exists(fname_normals) and osp.exists(fname_planes):
            return
        go_on = False
        try:
            fl = read_tiff(fname_normals)
            fl = read_tiff(fname_planes)
        except:
            go_on = True

        if go_on:
            print("Missing ", fname)
        else:
            return
    cmd = "./plane_normal_extraction/bin/segmentation '{}'".format(fname)
    os.popen(cmd).read()


def main():
    parser = argparse.ArgumentParser(
        description="Adds normals and smooth plannary surface segmentation \
                     to dataset with *.tiff depth files")
    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument("--calc_missing", action="store_true", default=False,
                        help="Find missing plane and normals files and generates them.")
    args = parser.parse_args()
    calc_missing = args.calc_missing
    if (not osp.exists(args.dataset)):
        print("Directory: {} does not exist. Skipping.".format(args.dataset))
        return
    filenames = sorted(glob.glob(args.dataset + '/**/*_depth_*.tiff', recursive=True))

    pool = Pool(2 * os.cpu_count() // 3)
    for _ in tqdm.tqdm(pool.imap_unordered(processDepth, filenames), total=len(filenames)):
        pass


if __name__ == '__main__':
    main()