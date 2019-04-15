import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(
            1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


class Depth2Points(nn.Module):
    def __init__(self, height, width):
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
        self.cos_theta = torch.cos(theta)
        self.sin_theta = torch.sin(theta)
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)

    def forward(self, depth, mask):
        X = depth * self.cos_phi * self.cos_theta * mask
        Y = depth * self.cos_phi * self.sin_theta * mask
        Z = depth * self.sin_phi * mask
        points = torch.cat((X, Y, Z), dim=1)
        return points


class SquaredGradientLoss(nn.Module):
    '''Compute the gradient magnitude of an image using the simple filters as in:
    Garg, Ravi, et al. "Unsupervised cnn for single view depth estimation: Geometry to the rescue."
    European Conference on Computer Vision. Springer, Cham, 2016.
    '''

    def __init__(self):

        super(SquaredGradientLoss, self).__init__()

        self.register_buffer('dx_filter', torch.FloatTensor([
            [0, 0, 0],
            [-0.5, 0, 0.5],
            [0, 0, 0]]).view(1, 1, 3, 3))
        self.register_buffer('dy_filter', torch.FloatTensor([
            [0, -0.5, 0],
            [0, 0, 0],
            [0, 0.5, 0]]).view(1, 1, 3, 3))

    def forward(self, pred, mask):
        dx = F.conv2d(
            pred,
            self.dx_filter.to(pred.get_device()),
            padding=1,
            groups=pred.shape[1])
        dy = F.conv2d(
            pred,
            self.dy_filter.to(pred.get_device()),
            padding=1,
            groups=pred.shape[1])

        error = mask * \
            (dx.abs().sum(1, keepdim=True) + dy.abs().sum(1, keepdim=True))

        return error.sum() / (mask > 0).sum().float()


class L2Loss(nn.Module):

    def __init__(self):

        super(L2Loss, self).__init__()

        self.metric = nn.MSELoss()

    def forward(self, pred, gt, mask):
        error = mask * self.metric(pred, gt)
        return error.sum() / (mask > 0).sum().float()


class MultiScaleL2Loss(nn.Module):

    def __init__(self, alpha_list, beta_list):

        super(MultiScaleL2Loss, self).__init__()

        self.depth_metric = L2Loss()
        self.grad_metric = SquaredGradientLoss()
        self.alpha_list = alpha_list
        self.beta_list = beta_list

    def forward(self, predictions, gt, mask_dict):

        # Go through each scale and accumulate errors
        depth_error = 0
        for i in range(len(predictions)):

            depth_pred, scale = predictions[i]
            depth_gt = gt[scale]
            mask = mask_dict[scale]
            alpha = self.alpha_list[i]
            beta = self.beta_list[i]

            # Compute depth error at this scale
            depth_error += alpha * self.depth_metric(
                depth_pred,
                depth_gt,
                mask)

            # Compute gradient error at this scale
            depth_error += beta * self.grad_metric(
                depth_pred,
                mask)

        return depth_error


class GradLoss(nn.Module):

    def __init__(self, all_levels=False):

        super(GradLoss, self).__init__()
        self.get_gradient = Sobel()
        self.l2_loss = L2Loss()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        self.all_levels = all_levels

    def createValidMean(self, mask, valid_pixels):
        def validMean(val):
            return (mask * val).sum() / valid_pixels
        return validMean

    def forward(self, predictions, gt_dict, mask_dict):
        total_loss = 0
        for i in range(len(predictions)):
            pred, scale = predictions[i]
            gt = gt_dict[scale]
            mask = mask_dict[scale]
            b, _, w, h = pred.size()
            # print(scale, pred.size(), gt.size(), mask.size())
            if i == 0 or self.all_levels:
                ones = pred.new_ones(b, 1, w, h)
                depth_grad = self.get_gradient(gt)
                output_grad = self.get_gradient(pred)

                depth_grad_dx = depth_grad[:, 0, :, :]
                depth_grad_dx = depth_grad_dx.contiguous().view_as(pred)

                depth_grad_dy = depth_grad[:, 1, :, :]
                depth_grad_dy = depth_grad_dy.contiguous().view_as(pred)

                output_grad_dx = output_grad[:, 0, :, :]
                output_grad_dx = output_grad_dx.contiguous().view_as(pred)

                output_grad_dy = output_grad[:, 1, :, :]
                output_grad_dy = output_grad_dy.contiguous().view_as(pred)

                depth_normal = torch.cat(
                    (-depth_grad_dx, -depth_grad_dy, ones), 1)
                output_normal = torch.cat(
                    (-output_grad_dx, -output_grad_dy, ones), 1)

                # depth_normal = F.normalize(depth_normal, p=2, dim=1)
                # output_normal = F.normalize(output_normal, p=2, dim=1)
                valid_pixels = (mask > 0).sum().float()
                validMean = self.createValidMean(mask, valid_pixels)

                loss_depth = validMean(torch.log(torch.abs(pred - gt) + 0.5))
                loss_dx = validMean(
                    torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5))
                loss_dy = validMean(
                    torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5))
                loss_normal = validMean(
                    torch.abs(1 - self.cos(output_normal, depth_normal)))
                total_loss += loss_depth + loss_normal + (loss_dx + loss_dy)
            else:
                total_loss += self.l2_loss(pred, gt, mask)

        return total_loss
