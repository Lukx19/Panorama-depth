import torch
import timeit
import functools
import argparse
import glob
import numpy as np

from visualizations import savePcl
from util import read_tiff, uncolapseMask
import unittest
import os.path as osp
from dataset import OmniDepthDataset
from modules import SmoothConv, Depth2Points
from network import PlanarToDepth
from annotated_data import DataType


class TestModules(unittest.TestCase):

    def setUp(self):
        self.dataset = OmniDepthDataset(root_path="../datasets/Omnidepth/",
                                        path_to_img_list="./test/test_split.txt",
                                        use_normals=True, use_planes=True)
        print(len(self.dataset))

    def savePointCloud(self, name, pts, rgb, normals, mask):
        mask = torch.reshape(mask, (-1, 1)).numpy()
        mask = np.squeeze(mask) == 1
        pts = pts[0].cpu().permute(1, 2, 0).numpy()
        pts = np.reshape(pts, (-1, 3))
        rgb = rgb.cpu()
        rgb = rgb.permute(1, 2, 0).numpy() * 250
        color = np.reshape(rgb, (-1, 3))
        normals = torch.squeeze(normals).permute(1, 2, 0).numpy()
        normals = np.reshape(normals, (-1, 3))
        savePcl(pts[mask, :], color[mask, :], osp.join('./test/', name + '.ply'),
                normals=normals[mask, :])

    def testSmoothConv(self):
        _, height, width = self.dataset[0][0][DataType.Depth].size()
        conv = SmoothConv(kernel_size=15, padding=7)
        to3d = Depth2Points(height, width)
        for i in range(len(self.dataset)):
            print(i)
            data = self.dataset[i][0]
            # basename = data = self.dataset[i][1]
            gt_depth = torch.reshape(data[DataType.Depth], (1, 1, height, width))
            normals = torch.reshape(data[DataType.Normals], (1, 3, height, width))
            planes = torch.reshape(data[DataType.Planes], (1, *data[DataType.Planes].size()))
            n_planes = planes.size(1)
            n_planes = 6
            rgb = data[DataType.Image]
            points = to3d(gt_depth)
            valid_mask = torch.sign(torch.sum(planes[:, 0:n_planes, :, :], dim=1))
            self.savePointCloud('gt_' + str(i), points, rgb, normals, valid_mask)
            res_cloud = torch.zeros_like(points)
            mask_comb = torch.zeros_like(valid_mask)
            for pl in range(3, n_planes):
                valid_mask = torch.sign(torch.sum(planes[:, pl:pl + 1, :, :], dim=1))
                smooth_pts = points.clone() * valid_mask
                normals_valid = valid_mask * normals.clone()
                for j in range(2):
                    smooth_pts = conv(smooth_pts, normals_valid.clone())
                self.savePointCloud('points3d_' + str(i) + "_" + str(pl), smooth_pts.clone(),
                                    rgb, normals, valid_mask)
                res_cloud = res_cloud + smooth_pts.clone()
                mask_comb = mask_comb + valid_mask
                self.savePointCloud('res_points3d_' + str(i) + "_" + str(pl), res_cloud.clone(),
                                    rgb, normals, mask_comb)

            valid_mask = torch.sign(torch.sum(planes[:, 0:n_planes, :, :], dim=1))
            self.savePointCloud('points3d_' + str(i), res_cloud, rgb, normals, valid_mask)
            # self.assertLess(torch.sum(torch.abs(smooth_pts - points)).item(), 1)

    # def testPlanar(self):

    #     _, height, width = self.dataset[0][0][DataType.Depth].size()
    #     planar_depth = PlanarToDepth(height=height, width=width)
    #     to3d = Depth2Points(height, width)
    #     for i in range(len(self.dataset)):
    #         data = self.dataset[i][0]
    #         # basename = data = self.dataset[i][1]
    #         gt_depth = torch.reshape(data[DataType.Depth], (1, 1, height, width))
    #         normals = torch.reshape(data[DataType.Normals], (1, 3, height, width))
    #         planes = torch.reshape(data[DataType.Planes], (1, *data[DataType.Planes].size()))
    #         valid_mask = torch.sign(torch.sum(planes[:, 0:4, :, :, :], dim=1))
    #         rgb = data[DataType.Image]
    #         depth_planar = planar_depth(gt_depth, planes, None, normals)
    #         points = to3d(depth_planar)
    #         self.savePointCloud('planar_' + str(i), points, rgb, normals)

            # self.assertLess(torch.sum(torch.abs(smooth_pts - points)).item(), 1)

# class TestUtilMethods(unittest.TestCase):

#     def testMaskColapsing(self):
#         mask = np.array([[[0, 1], [2, 2], [3, 4]], [[1], [1]]])
#         mask = torch.from_numpy(mask)
#         res = uncolapseMask(mask)
#         self.assertEqual(len(res), 2)
#         self.assertEqual(res[0].size(), (1, 5, 3, 2))
#         self.assertEqual(res[1].size(), (1, 1, 1, 1))
#         res0 = res[0][0].numpy()
#         self.assertEqual(res0[0], np.array([[1, 0], [0, 0], [0, 0]]))
#         self.assertEqual(res0[1], np.array([[0, 1], [0, 0], [0, 0]]))
#         self.assertEqual(res0[2], np.array([[0, 0], [1, 1], [0, 0]]))
#         self.assertEqual(res0[4], np.array([[0, 0], [0, 0], [1, 0]]))
#         self.assertEqual(res0[5], np.array([[0, 0], [0, 0], [0, 1]]))


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
    unittest.main(verbosity=3)
    # testPanoToPointsConversion()
