import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Depth2Points(nn.Module):
    def __init__(self, height, width):
        super(Depth2Points, self).__init__()
        phi = torch.zeros((height, width))
        theta = torch.zeros((height, width))

        hcam_deg = 360
        vcam_deg = 180
        # Camera rotation angles in radians
        hcam_rad = hcam_deg / 180.0 * np.pi
        vcam_rad = vcam_deg / 180.0 * np.pi
        # print(hcam_deg, vcam_deg)
        for v in range(height):
            for u in range(width):
                theta[v, u] = (u - width / 2.0) / width * hcam_rad
                phi[v, u] = -(v - height / 2.0) / height * vcam_rad
        self.cos_theta = nn.Parameter(torch.cos(theta), requires_grad=False)
        self.sin_theta = nn.Parameter(torch.sin(theta), requires_grad=False)
        self.cos_phi = nn.Parameter(torch.cos(phi), requires_grad=False)
        self.sin_phi = nn.Parameter(torch.sin(phi), requires_grad=False)

    def forward(self, depth):
        X = depth * self.cos_phi * self.cos_theta
        Y = depth * self.cos_phi * self.sin_theta
        Z = depth * self.sin_phi
        points = torch.cat((X, Y, Z), dim=1)
        # print(points)
        return points


class SmoothConv(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=0, iterations=1):
        super(SmoothConv, self).__init__()
        if type(kernel_size) is int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation,
                                      padding=padding, stride=1)
        self.iterations = iterations

    def normalDist(self, x, sigma):
        return torch.exp((-x**2) / (2 * sigma**2))

    def forward(self, points, normals, segmentation=None):
        _, _, height, width = points.size()
        if segmentation is None:
            combined_data = torch.cat((points, normals.clone()), dim=1)
        else:
            combined_data = torch.cat((points, normals.clone(), segmentation), dim=1)

        unfolded = self.unfold(combined_data)
        b, n, k = unfolded.size()
        Kh, Kw = self.kernel_size
        kernel_mid = Kw * (Kh // 2) + Kw // 2

        unfolded = torch.transpose(unfolded, dim0=1, dim1=2)
        unfolded = torch.reshape(unfolded, (b, k, -1, Kh * Kw))
        if segmentation is None:
            pts_ker, normals_ker = torch.split(unfolded, [3, 3], dim=2)
        else:
            pts_ker, normals_ker, seg_ker = torch.split(unfolded, [3, 3, 1], dim=2)

        pts_i = pts_ker[:, :, :, kernel_mid:kernel_mid + 1]
        normals_i = normals_ker[:, :, :, kernel_mid:kernel_mid + 1]

        # Caluclate weightin factor
        points_diff = pts_ker - pts_i
        pts_norm = torch.norm(points_diff, p=2, dim=2, keepdim=True)
        sigma_c, _ = torch.max(pts_norm, dim=3, keepdim=True)
        # print(pts_norm.size(), sigma_c.size())
        sigma_c = sigma_c / 5 + 1e-5
        W_p = self.normalDist(pts_norm, sigma_c)

        sigma_s = 1.0 / 3.0
        cos_angles = torch.sum(normals_i * normals_ker, dim=2, keepdim=True)
        W_n = self.normalDist(1 - cos_angles, sigma_s)
        if segmentation is None:
            weight = W_n * W_p
        else:
            W_s = seg_ker
            weight = W_n * W_p * W_s
        # weight = W_c
        weight = weight.clone().detach()
        normalization = torch.sum(weight.clone(), dim=3)
        mask = torch.sign(torch.abs(pts_i))
        # print(weight.size())
        for it in range(self.iterations):
            points_diff = pts_ker - pts_i
            normal_pts_dot = weight * torch.sum(points_diff * normals_ker, dim=2, keepdim=True)
        # normal_pts_dot = weight
        # improvement = (torch.sum(normals_ker * normal_pts_dot, dim=3)) / (Kh * Kw)
            improvement = (torch.sum(normals_ker * normal_pts_dot, dim=3)) / normalization
        # print(improvement.size())
            improvement = improvement.permute(0, 2, 1)
            improvement = torch.reshape(improvement, (b, 3, height, width))

            if self.iterations > 1:
                unfolded_imp = self.unfold(improvement)
                unfolded_imp = torch.transpose(unfolded_imp, dim0=1, dim1=2)
                unfolded_imp = torch.reshape(unfolded_imp, (b, k, -1, Kh * Kw))
                pts_ker = (pts_ker + unfolded_imp) * mask
                pts_i = pts_ker[:, :, :, kernel_mid:kernel_mid + 1]

        if self.iterations == 1:
            points = points + improvement
        else:
            points = torch.squeeze(pts_i).permute(0, 2, 1)
            points = torch.reshape(points, (b, 3, height, width))
        return points


class PlanarConv(nn.Module):
    def __init__(self, height, width, kernel_size):
        super(PlanarConv, self).__init__()
        half_kernel = kernel_size // 2
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=1,
                                      padding=half_kernel, stride=kernel_size)
        self.fold = torch.nn.Fold(output_size=(height, width), kernel_size=kernel_size,
                                  dilation=1, padding=half_kernel, stride=kernel_size)
        self.height = height
        self.width = width
        self.to3d = Depth2Points(height, width)
        if type(kernel_size) is int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

    def forward(self, depth, normals):
        _, _, height, width = depth.size()
        points = self.to3d(depth)
        rays = self.to3d(torch.ones_like(depth))
        # norm_pts = torch.sum(normals * points, dim=1, keepdim=True)
        unfolded = self.unfold(torch.cat((points, rays, normals), dim=1))
        # print(unfolded.size())
        b, n, k = unfolded.size()
        Kh, Kw = self.kernel_size
        unfolded = torch.transpose(unfolded, dim0=1, dim1=2)
        unfolded = torch.reshape(unfolded, (b, k, 9, Kh * Kw))
        pts_ker, rays_ker, normals_ker = torch.split(unfolded, [3, 3, 3], dim=2)

        avg_normal = torch.mean(normals_ker, dim=3, keepdim=True)
        centroids = torch.mean(pts_ker, dim=3, keepdim=True)
        # print(pts_ker.size(), centroids.size())
        # Calculates weights between average normal and the rest of normals in kernel.
        # If are normals similar resulting weight will be 1.
        normal_similarity = torch.sum(avg_normal * normals_ker, dim=2, keepdim=True)
        # TODO: add flip mask if normal is estimate in the wrong direction
        # normal_similarity = torch.clamp(normal_similarity, min=0)
        # print("minmax normal sim: ", normal_similarity.size(),
        #   torch.min(normal_similarity), torch.max(normal_similarity))

        norm = torch.sum(rays_ker * (avg_normal.detach()), dim=2, keepdim=True)
        # print(norm.size(), planes_r.size(), rays.size(), torch.min(norm))
        norm = norm + (1 - torch.abs(torch.sign(norm))) * 1e-5

        centroid_normals = normal_similarity * centroids * avg_normal
        planar_depths = torch.sum(centroid_normals, dim=2, keepdim=True)
        planar_depths = planar_depths / norm

        nan_count = (1 - torch.isfinite(planar_depths)).sum().item()
        if nan_count > 0:
            print("found # inifinst", nan_count)

        planar_depths = torch.where(torch.isfinite(planar_depths),
                                    planar_depths, torch.zeros_like(planar_depths))

        planar_depths = torch.reshape(planar_depths, (b, k, -1))
        planar_depths = torch.transpose(planar_depths, dim0=1, dim1=2)
        folded_depth = self.fold(planar_depths)
        return folded_depth


class Points2Depths(nn.Module):
    def __init__(self, height, width):
        super(Points2Depths, self).__init__()
        self.height = height
        self.width = width
        self.to3d = Depth2Points(height, width)

    def forward(self, depth, points, normals=None):
        b, ch, h, w = depth.size()
        points = torch.reshape(points, (b, 3, -1))
        original_pts = self.to3d(depth)
        original_pts = torch.reshape(original_pts, (b, 3, -1))
        rays = self.to3d(torch.ones_like(depth))
        rays = torch.reshape(rays, (b, 3, -1))
        if normals is None:
            vec_a = points - original_pts
            cos_angle = F.cosine_similarity(rays, vec_a, dim=1)
            cos_angle = torch.max(cos_angle, torch.ones_like(cos_angle) * 1e-5)
            cos_angle = torch.reshape(cos_angle, (b, 1, -1))
            print(cos_angle.min(), cos_angle.max(), cos_angle[cos_angle > 1e-4].median())
            magnitute_a = torch.norm(vec_a, p=2, dim=1, keepdim=True)
            print(magnitute_a.min(), magnitute_a.max(), magnitute_a[magnitute_a > 0].median())

            improvement = magnitute_a / cos_angle
            print(improvement.min(), improvement.max(), improvement[improvement > 0].median())
            improvement = torch.reshape(improvement, (b, 1, h, w))
            depth = depth + improvement
        else:
            normals = torch.reshape(normals, (b, 3, -1))
            norm = torch.sum(rays * (normals.detach()), dim=1, keepdim=True)
            norm = norm + (1 - torch.abs(torch.sign(norm))) * 1e-5

            centroid_normals = points * normals
            planar_depth = torch.sum(centroid_normals, dim=1, keepdim=True)
            planar_depth = planar_depth / norm
            planar_depth = torch.reshape(planar_depth, (b, 1, h, w))
            depth_diff = torch.abs(depth - planar_depth)
            depth = torch.where(depth_diff > 0.8, depth, planar_depth)

        return depth
