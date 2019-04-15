import torch
import timeit
import functools
import argparse
import glob
import numpy as np
from criteria import Depth2Points
from visualizations import savePcl
from util import read_tiff


def testPanoToPointsConversion():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', type=str, default="../datasets/test/",
                        help='Dataset storage folder')

    args = parser.parse_args()
    results_dir = args.results_folder
    regex = "*_depth_*.tiff"

    depths = sorted(glob.glob(results_dir + "/**/" + regex, recursive=True))
    print(depths)
    for depth_path in depths:
        depth = np.squeeze(read_tiff(depth_path))
        filename = depth_path[:-4] + "ply"

        print(filename)
        height, width = depth.shape

        converter = Depth2Points(height, width).forward
        depth = torch.from_numpy(depth)
        depth = depth.reshape(1, 1, height, width)
        mask = depth > 8
        mask += depth < 0.001
        mask = 1 - torch.sign(mask).float()
        print(mask.size(), torch.sum(mask))
        t = timeit.Timer(functools.partial(converter, depth, mask))
        print(t.timeit(10))

        uvdepths = converter(depth, mask)
        print(uvdepths.size())
        pcd = uvdepths.permute(0, 2, 3, 1)
        print(pcd.size())
        pcd = pcd.view(-1, 3)
        print(pcd.size())

        savePcl(pcd, [], filename)


if __name__ == "__main__":
    testPanoToPointsConversion()
