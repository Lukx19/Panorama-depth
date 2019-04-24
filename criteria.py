import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from annotated_data import DataType


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
            [0, 0, 0]]).reshape(1, 1, 3, 3))
        self.register_buffer('dy_filter', torch.FloatTensor([
            [0, -0.5, 0],
            [0, 0, 0],
            [0, 0.5, 0]]).reshape(1, 1, 3, 3))

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


def createValidMean(mask):
    valid_pixels = (mask > 0).sum().float()

    def validMean(val):
        return (mask * val).sum() / valid_pixels
    return validMean


def cosineLoss(pred_normal, gt_normal, mean_fce=torch.mean):
    return mean_fce(
        torch.abs(1 - F.cosine_similarity(pred_normal, gt_normal, dim=1, eps=1e-8)))


def L1Loss(pred, target, mean_fce=torch.mean):
    val = F.l1_loss(pred, target)
    return mean_fce(val)

# https://github.com/kmaninis/OSVOS-PyTorch


def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = label.float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()

    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= int(np.prod(label.size()))
    elif batch_average:
        final_loss /= int(label.size(0))

    return final_loss
    #  ----------------------------------------------------------


class MultiScaleL2Loss(nn.Module):

    def __init__(self, alpha_list, beta_list):

        super(MultiScaleL2Loss, self).__init__()

        self.depth_metric = L2Loss()
        self.grad_metric = SquaredGradientLoss()
        self.alpha_list = alpha_list
        self.beta_list = beta_list

    def forward(self, predictions, data):

        # Go through each scale and accumulate errors
        depth_error = 0
        for i, (scale, depth_pred) in enumerate(predictions.queryType(DataType.Depth)):

            depth_gt = data.get(DataType.Depth, scale=scale)[0]
            mask = data.get(DataType.Mask, scale=scale)[0]
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
        self.all_levels = all_levels

    def forward(self, predictions, data):
        total_loss = 0
        for i, (scale, pred) in enumerate(predictions.queryType(DataType.Depth)):

            gt = data.get(DataType.Depth, scale=scale)[0]
            mask = data.get(DataType.Mask, scale=scale)[0]
            b, _, w, h = pred.size()
            # print(scale, pred.size(), gt.size(), mask.size())
            if i == 0 or self.all_levels:
                ones = pred.new_ones(b, 1, w, h)

                # gt += (1 - mask) * 0.0001
                # pred += (1 - mask) * 0.0001

                # gt = torch.log(gt)
                # pred = torch.log(pred)

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
                validMean = createValidMean(mask)

                loss_depth = validMean((torch.abs(pred - gt)))
                loss_dx = validMean(
                    (torch.abs(output_grad_dx - depth_grad_dx)))
                loss_dy = validMean(
                    (torch.abs(output_grad_dy - depth_grad_dy)))
                loss_normal = cosineLoss(output_normal, depth_normal, validMean)
                # print(loss_depth, loss_normal, loss_dx, loss_dy)
                total_loss += loss_depth + loss_normal + (loss_dx + loss_dy)
            else:
                total_loss += self.l2_loss(pred, gt, mask)

        return total_loss


class NormSegLoss(nn.Module):
    def __init__(self):
        super(NormSegLoss, self).__init__()
        self.depth_loss = GradLoss(all_levels=True)
        # self.normal_loss =
        self.segmentation_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, data):
        depth_total_loss = self.depth_loss(predictions, data)

        mask = data.get(DataType.Mask, scale=1)[0]
        gt_seg = data.get(DataType.PlanarSegmentation)[0] * mask
        pred_seg = predictions.get(DataType.PlanarSegmentation)[0] * mask

        validMean = createValidMean(mask)

        # seg_total_loss = self.segmentation_loss(pred_seg, gt_seg)
        seg_total_loss = class_balanced_cross_entropy_loss(pred_seg, gt_seg)

        gt_normals = data.get(DataType.Normals)[0] * mask
        pred_normals = predictions.get(DataType.Normals)[0] * mask
        normals_loss = cosineLoss(pred_normals, gt_normals, validMean)
        return depth_total_loss + seg_total_loss + normals_loss


class PlaneNormSegLoss(nn.Module):
    def __init__(self, width=256, height=512):
        super(PlaneNormSegLoss, self).__init__()
        self.depth_loss = GradLoss(all_levels=True)
        # self.normal_loss =
        self.segmentation_loss = nn.BCEWithLogitsLoss()
        self.to3d = Depth2Points(width, height)

    def averageNormals(self, normals, planes):
        avg_plane_normal = torch.mean(planes * normals, dim=3)
        norm = torch.norm(avg_plane_normal, p=None, keepdim=True, dim=2)
        mask = norm.detach() < 0.0001
        norm = norm + torch.ones_like(norm) * mask.float()
        return avg_plane_normal / norm

    def forward(self, predictions, data):
        depth_total_loss = self.depth_loss(predictions, data)
        mask = data.get(DataType.Mask, scale=1)[0]

        gt_seg = data.get(DataType.PlanarSegmentation)[0] * mask
        pred_seg = predictions.get(DataType.PlanarSegmentation)[0] * mask
        pred_depth = predictions.get(DataType.Depth, scale=1)[0]
        gt_depth = data.get(DataType.Depth, scale=1)[0] * mask
        gt_normals = data.get(DataType.Normals)[0] * mask
        pred_normals = predictions.get(DataType.Normals)[0] * mask

        b, ch, h, w = pred_normals.size()
        # seg_total_loss = self.segmentation_loss(pred_seg, gt_seg)
        validMean = createValidMean(mask)
        seg_total_loss = class_balanced_cross_entropy_loss(pred_seg, gt_seg)
        similarity = cosineLoss(pred_normals, gt_normals, validMean)

        pred_points = self.to3d(pred_depth)
        with torch.no_grad():
            gt_points = self.to3d(gt_depth)

        # TODO: Use ground truth points which are distance to plane and not GT distance in general.
        #  This should improve performance on low quality GT data
        distace_to_gt_plane = (pred_points - gt_points) * gt_normals
        distace_to_pred_plane = (gt_points - pred_points) * pred_normals
        distance_loss = L1Loss(distace_to_gt_plane, torch.zeros_like(distace_to_gt_plane))
        distance_loss2 = L1Loss(-distace_to_pred_plane, torch.zeros_like(distace_to_pred_plane))

        #  This losses look at all normals of one plane and create plane parameters
        planes3, avg_gt_normal = None, None
        pred_points = torch.reshape(pred_points, (b, 1, 3, -1))
        gt_points = torch.reshape(gt_points, (b, 1, 3, -1))
        gt_normals = torch.reshape(gt_normals, (b, 1, 3, -1))
        pred_normals = torch.reshape(pred_normals, (b, 1, 3, -1))
        with torch.no_grad():
            planes = data.get(DataType.Planes, scale=1)[0]
            planes = planes * mask
            b, p, h, w = planes.size()
            planes3 = torch.reshape(planes, (b, p, 1, -1))
            planes3 = torch.cat((planes3, planes3, planes3), dim=2)
            avg_gt_normal = self.averageNormals(gt_normals, planes3)

        avg_pred_normal = self.averageNormals(pred_normals, planes3)
        similarity_planar = cosineLoss(avg_gt_normal, avg_pred_normal)

        distance_pred_planes = planes3 * (gt_points - pred_points)

        distance_pred_planes = distance_pred_planes * torch.reshape(avg_pred_normal, (b, p, 3, 1))

        distance_loss_plane = L1Loss(-distance_pred_planes,
                                     torch.zeros_like(distance_pred_planes))

        return {
            "Depth_Loss": depth_total_loss,
            "Segmentation_Loss": seg_total_loss,
            "Normal_Cosine_Loss": similarity,
            "Planar_Cosine_Loss": similarity_planar,
            "Plane_Distance_Loss": distance_loss,
            "Plane_Distance_Loss2": distance_loss2,
            "Distance_Pred_Plane_Loss": distance_loss_plane,
            # "Plane_Loss": plane_loss,
            # "L1_Normal_Loss": normals_loss
        }
