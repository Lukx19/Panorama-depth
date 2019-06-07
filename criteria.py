import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from annotated_data import DataType
from network import Depth2Points, planeAwarePooling, toPlaneParams, signToLabel, DepthToNormals
from util import mergeInDict


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


def createValidMean(mask, adaptive=False):
    valid_pixels = (mask > 0).sum().float()

    def validMean(val):
        # print(mask.size(), val.size())
        count_pixels = valid_pixels
        val = mask * val
        if adaptive:
            with torch.no_grad():
                max_p, _ = torch.max(val, dim=1, keepdim=True)
                tresholds = 0.75 * max_p
                error_mask = torch.zeros_like(val)
                error_mask[val > tresholds] = 1
                count_pixels = torch.sum(error_mask)

        val = torch.sum(val)
        # print(val.size())
        return val / (count_pixels)

    return validMean


def createPlaneMean(mask: torch.Tensor, planes, adaptive=False):
    """Creates a function which can calculate a mean over plane instance.

    Parameters
    ----------
    mask : Tensor
        pixels excluded from loss calculation
    planes : Tensor BxPx...
        Plane instance masks
    adaptive : bool, optional
        Calculates mean only over the most loss heavy pixels, by default False

    Returns
    -------
    function
        Mean function
    """
    b, p = planes.size()[0:2]
    # valid_pixels = torch.sum(planes * mask)
    planes = torch.reshape(planes, (b, p, -1))
    # B x P
    valid_per_plane = torch.sum(planes.detach(), dim=2)

    valid_planes_in_batches = torch.sum(torch.sign(valid_per_plane))
    # #  avoid division by 0
    valid_per_plane[valid_per_plane < 0.0001] = 1

    def mean(val):
        count_per_plane = valid_per_plane
        count_planes_in_batches = valid_planes_in_batches
        val = torch.reshape(val, (b, 1, -1))
        planar_val = planes * val
        # B x P
        if adaptive:
            with torch.no_grad():
                max_p, _ = torch.max(planar_val, dim=2, keepdim=True)
                tresholds = 0.75 * max_p
                error_mask = torch.zeros_like(planar_val)
                error_mask[planar_val > tresholds] = 1

                count_per_plane = torch.sum(error_mask, dim=2)
                count_planes_in_batches = torch.sum(torch.sign(valid_per_plane))
                count_per_plane[count_per_plane < 0.0001] = 1
            planar_val = planar_val * error_mask
            summed = torch.sum(planar_val, dim=2)
        else:
            summed = torch.sum(planar_val, dim=2)

        return torch.sum(summed / count_per_plane) / count_planes_in_batches

    return mean


def cosineLoss(pred_normal, gt_normal, dim=1, mean_fce=torch.mean):
    val = torch.abs(1 - F.cosine_similarity(pred_normal, gt_normal, dim=dim, eps=1e-8))
    if mean_fce is None:
        return val
    return mean_fce(val)


# def cosineLossNorm(pred, gt):
#     '''
#         pred, gt: BxPx3xN
#     '''
#     pred_norm = torch.norm(pred, p=None, keepdim=True, dim=2)
#     gt_norm = torch.norm(gt, p=None, keepdim=True, dim=2)
#     return cosineLoss(pred / pred_norm, gt / gt_norm, dim=2, mean_fce=None)


def l1Loss(pred, target, mean_fce=None, dim=1):
    val = F.l1_loss(pred, target, reduction='none')
    if mean_fce is None:
        return val
    return mean_fce(val)


def l2Loss(pred, target, mean_fce=None):
    val = F.mse_loss(pred, target, reduction='none')
    if mean_fce is None:
        return val
    # print(val.size())
    return mean_fce(val)


def huberLoss(pred, target, mask):
    x = pred - target
    x = x * mask
    abs_val = torch.abs(x)
    treshold = 1 / 5 * torch.max(abs_val)
    treshold_mask = (abs_val <= treshold).float()
    below_tresh = abs_val * treshold_mask
    above_tresh = (1 - treshold_mask) * ((x**2 + treshold**2) / (2 * treshold))
    return below_tresh + above_tresh

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


def revisLoss(pred, gt, mask, sobel, adaptive=False):
    b, _, h, w = pred.size()
    ones = pred.new_ones(b, 1, h, w)
    histograms = {}

    depth_grad = sobel(gt)
    output_grad = sobel(pred)

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

    validMean = createValidMean(mask, adaptive=False)
    #  L1 loss on depth distances
    loss_depth = torch.log(l1Loss(pred, gt) + 1)
    histograms["revis_l1_dist"] = loss_depth.detach()
    loss_depth = validMean(loss_depth)

    validMean = createValidMean(mask, adaptive=adaptive)
    loss_dx = torch.log(l1Loss(output_grad_dx, depth_grad_dx) + 1)
    loss_dy = torch.log(l1Loss(output_grad_dy, depth_grad_dy) + 1)
    histograms["revis_dxdy"] = (loss_dx + loss_dy).detach()
    loss_dx = validMean(loss_dx)
    loss_dy = validMean(loss_dy)
    loss_normal = cosineLoss(output_normal, depth_normal, dim=1, mean_fce=None)
    histograms["revis_normal_similarity"] = loss_normal.detach()
    loss_normal = validMean(loss_normal)
    # print(loss_depth, loss_normal, loss_dx, loss_dy)
    losses = {
        "revis_l1_dist": loss_depth,
        "revis_dxdy": loss_normal,
        "revis_normal_similarity": loss_dx + loss_dy
    }
    return losses, histograms


def sumLosses(losses):
    total_loss = 0
    for key, loss in losses.items():
        total_loss += loss
    return total_loss


class GradLoss(nn.Module):

    def __init__(self, height=256, width=512, all_levels=False,
                 adaptive=False, depth_normals=False):

        super(GradLoss, self).__init__()
        self.sobel = Sobel()
        self.l2_loss = L2Loss()
        self.all_levels = all_levels
        self.adaptive = adaptive
        self.depth_normals = depth_normals
        if self.depth_normals:
            self.depth_cosine_loss = CosineDepthNormalsLoss(height, width)

    def forward(self, predictions, data):
        losses = {}
        histograms = {}
        for i, (scale, pred) in enumerate(predictions.queryType(DataType.Depth)):
            gt = data.get(DataType.Depth, scale=scale)[0]
            mask = data.get(DataType.Mask, scale=scale)[0]
            if i == 0 or self.all_levels:
                revis_losses, revis_hist = revisLoss(pred, gt, mask, self.sobel, self.adaptive)
                for key, loss in revis_losses.items():
                    losses[key + "_" + str(scale) + "_" + str(i)] = loss

                for key, hist in revis_hist.items():
                    histograms[key + "_" + str(scale) + "_" + str(i)] = hist
            else:
                l2_loss = self.l2_loss(pred, gt, mask)
                losses["revis_l2" + "_" + str(scale) + "_" + str(i)] = l2_loss

        if self.depth_normals:
            l, h = self.depth_cosine_loss(predictions, data)
            losses = mergeInDict(losses, l)
            histograms = mergeInDict(histograms, h)

        return losses, histograms


class NormSegLoss(nn.Module):
    def __init__(self):
        super(NormSegLoss, self).__init__()
        self.depth_loss = GradLoss(all_levels=True)
        # self.normal_loss =
        self.segmentation_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, data):
        depth_total_loss = self.depth_loss(predictions, data)

        mask = data.get(DataType.Mask, scale=1)[0]
        gt_seg = data.get(DataType.PlanarSegmentation)[0]
        pred_seg = predictions.get(DataType.PlanarSegmentation)[0]

        validMean = createValidMean(mask)

        # seg_total_loss = self.segmentation_loss(pred_seg, gt_seg)
        seg_total_loss = class_balanced_cross_entropy_loss(pred_seg, gt_seg)

        gt_normals = data.get(DataType.Normals)[0] * mask
        pred_normals = predictions.get(DataType.Normals)[0] * mask
        normals_loss = cosineLoss(pred_normals, gt_normals, dim=1, mean_fce=validMean)
        return depth_total_loss + seg_total_loss + normals_loss


class SphericalNormalsLoss(nn.Module):
    def __init__(self):
        super(SphericalNormalsLoss, self).__init__()

    def forward(self, predictions, data):
        losses = {}
        histograms = {}
        planes = data.get(DataType.Planes, scale=1)[0]
        # 1= pixels where are some planes 0 = otherwise
        # plane_mask = torch.sign(torch.sum(planes, dim=1))
        mask = data.get(DataType.Mask, scale=1)[0]
        gt_normals = data.get(DataType.Normals)[0]
        pred_normals_emb = predictions.get(DataType.NormalsEmbed)[0] * mask
        pred_classes = predictions.get(DataType.NormalsClass)[0]
        gt_normals_emb = torch.abs(gt_normals) * mask
        gt_classes = signToLabel(gt_normals)

        planeMean = createPlaneMean(mask, planes)

        # pred_normals are in range [0,1] calculate cosine similarity
        regression_loss = cosineLoss(pred_normals_emb, gt_normals_emb, dim=1, mean_fce=None)
        histograms["plane_normal_regression"] = regression_loss.detach()
        regression_loss = planeMean(regression_loss)
        losses["plane_normal_regression"] = regression_loss

        # classification of normal signs
        class_loss = F.cross_entropy(pred_classes, gt_classes, reduction='none')
        histograms["plane_normal_class"] = class_loss.detach()
        losses["plane_normal_class"] = planeMean(class_loss)
        return losses, histograms


class CosineNormalsLoss(nn.Module):
    def __init__(self):
        super(CosineNormalsLoss, self).__init__()

    def forward(self, predictions, data):
        histograms = {}
        losses = {}
        mask = data.get(DataType.Mask, scale=1)[0]
        # middle_mask = torch.zeros_like(mask)
        # middle_mask[:, :, :, 138:370] = 1
        planes = data.get(DataType.Planes, scale=1)[0]
        # 1= pixels where are some planes 0 = otherwise
        plane_mask = torch.sign(torch.sum(planes, dim=1))
        gt_normals = data.get(DataType.Normals)[0] * mask
        pred_normals = predictions.get(DataType.Normals)[0] * mask

        b, _, h, w = pred_normals.size()
        validMean = createValidMean(mask)
        planeMean = createPlaneMean(mask, planes)
        similarity = cosineLoss(pred_normals, gt_normals, dim=1, mean_fce=None)
        similarity *= plane_mask
        histograms["Pixel_normal_similarity_loss"] = similarity.detach()
        losses["Pixel_normal_similarity_loss"] = validMean(similarity)
        losses["Plane_normal_similarity_loss"] = planeMean(similarity)

        # plane_mask_ext = torch.reshape(plane_mask, (b, 1, h, w))
        # pred_normals_2x = F.avg_pool2d(pred_normals * plane_mask_ext, kernel_size=3,
        #                                padding=1, stride=2)
        # with torch.no_grad():
        #     gt_normals_2x = F.avg_pool2d(gt_normals * plane_mask_ext, kernel_size=3,
        #                                  padding=1, stride=2)
        #     plane_mask_2x = F.avg_pool2d(plane_mask, kernel_size=3, padding=1, stride=2)

        # validMean2x = createValidMean(plane_mask_2x)
        # similarity_2x = cosineLoss(pred_normals_2x, gt_normals_2x, dim=1, mean_fce=None)
        # similarity_2x *= plane_mask_2x
        # # histograms["Pixel_normal_similarity_2x"] = similarity_2x.detach()
        # losses["Pixel_normal_similarity_2x"] = validMean2x(similarity_2x)

        return losses, histograms


class CosineDepthNormalsLoss(nn.Module):

    def __init__(self, height, width):
        super(CosineDepthNormalsLoss, self).__init__()
        self.depth_normals = DepthToNormals(height=height, width=width,
                                            kernel_size=3, padding=2, dilation=2)

    def forward(self, predictions, data):
        histograms = {}
        losses = {}
        mask = data.get(DataType.Mask, scale=1)[0]
        gt_depth = data.get(DataType.Depth)[0] * mask

        pred_normals = predictions.get(DataType.DepthNormals)[0] * mask

        # pred_depth = predictions.get(DataType.Depth)[0] * mask
        # pred_normals = self.depth_normals(pred_depth)

        with torch.no_grad():
            gt_normals = self.depth_normals(gt_depth)

        validMean = createValidMean(mask)
        similarity = cosineLoss(pred_normals, gt_normals, dim=1, mean_fce=None)
        histograms["Pixel_depth_normal_similarity_loss"] = similarity.detach()
        losses["Pixel_depth_normal_similarity_loss"] = validMean(similarity)

        return losses, histograms


class PlaneNormSegLoss(nn.Module):
    def __init__(self, width=512, height=256, normal_loss=CosineNormalsLoss()):
        super(PlaneNormSegLoss, self).__init__()
        self.to3d = Depth2Points(height, width)
        # self.sobel = Sobel()
        self.normal_loss = normal_loss
        self.grad_loss = GradLoss(all_levels=True, adaptive=False, depth_normals=False)
        # self.depth_cosine_loss = CosineDepthNormalsLoss(height, width)

    def averageNormals(self, normals, planes):
        avg_plane_normal = torch.mean(planes * normals, dim=3)
        norm = torch.norm(avg_plane_normal, p=None, keepdim=True, dim=2)
        mask = norm.detach() < 0.0001
        norm = norm + torch.ones_like(norm) * mask.float()
        return avg_plane_normal / norm

    def forward(self, predictions, data):
        # losses = {}
        # histograms = {}
        planes = data.get(DataType.Planes, scale=1)[0]
        # 1= pixels where are some planes 0 = otherwise
        plane_mask = torch.sign(torch.sum(planes, dim=1))
        losses, histograms = self.grad_loss(predictions, data)

        # l, h = self.depth_cosine_loss(predictions, data)
        # losses = mergeInDict(losses, l)
        # histograms = mergeInDict(histograms, h)
        # for i, (scale, pred) in enumerate(predictions.queryType(DataType.Depth)):
        #     gt = data.get(DataType.Depth, scale=scale)[0]
        #     mask = data.get(DataType.Mask, scale=scale)[0]
        #     revis_losses, revis_hist = revisLoss(pred, gt, mask, self.sobel)
        #     for key, loss in revis_losses.items():
        #         losses[key + "_" + str(scale)] = loss

        #     for key, hist in revis_hist.items():
        #         histograms[key + "_" + str(scale)] = hist

        mask = data.get(DataType.Mask, scale=1)[0]

        gt_seg = data.get(DataType.PlanarSegmentation)[0] * mask
        pred_seg = predictions.get(DataType.PlanarSegmentation)[0] * mask
        pred_depth = predictions.get(DataType.Depth, scale=1)[0]
        gt_depth = data.get(DataType.Depth, scale=1)[0] * mask
        gt_normals = data.get(DataType.Normals)[0] * mask
        pred_normals = predictions.get(DataType.Normals)[0] * mask
        # print(torch.sum(plane_mask), torch.sum(gt_seg), torch.sum(mask))
        b, ch, h, w = pred_normals.size()

        seg_total_loss = class_balanced_cross_entropy_loss(pred_seg, gt_seg)
        losses["Segmentation_Loss"] = seg_total_loss

        # validMean = createValidMean(plane_mask)
        planeMean = createPlaneMean(mask, planes)

        pred_points = self.to3d(pred_depth)
        with torch.no_grad():
            gt_points = self.to3d(gt_depth)

        # TODO: Use ground truth points which are distance to plane and not GT distance in general.
        #  This should improve performance on low quality GT data
        distance = (pred_points - gt_points) * gt_normals
        distace_to_gt_plane = (torch.sum(distance, dim=1))
        distace_to_gt_plane = distace_to_gt_plane ** 2

        distace_to_gt_plane *= plane_mask

        histograms["Pixel_dist_plane_loss"] = distace_to_gt_plane.detach()
        # losses["Pixel_dist_plane_loss"] = validMean(distace_to_gt_plane)

        losses["Plane_dist_plane_loss"] = planeMean(distace_to_gt_plane)
        planeMean = createPlaneMean(mask, planes, adaptive=True)
        losses["Plane_dist_plane_loss_Ad"] = planeMean(distace_to_gt_plane)

        # distance_loss2 = l1Loss(-distace_to_pred_plane,
        # torch.zeros_like(distace_to_pred_plane), validMean)

        #  This losses look at all normals of one plane and create plane parameters
        avg_gt_normal = None
        pred_points = torch.reshape(pred_points, (b, 1, 3, -1))
        gt_points = torch.reshape(gt_points, (b, 1, 3, -1))
        gt_normals = torch.reshape(gt_normals, (b, 1, 3, -1))
        pred_normals = torch.reshape(pred_normals, (b, 1, 3, -1))
        with torch.no_grad():
            planes = data.get(DataType.Planes, scale=1)[0]
            planes = planes * mask
            b, p, h, w = planes.size()
            planes = torch.reshape(planes, (b, p, 1, -1))
            # planes_ext = torch.cat((planes, planes, planes), dim=2)
            avg_gt_normal = planeAwarePooling(gt_normals, planes)

        avg_pred_normal = planeAwarePooling(pred_normals, planes)
        similarity_planar = cosineLoss(avg_gt_normal, avg_pred_normal, dim=2, mean_fce=None)
        # histograms["Plane_normal_loss"] = similarity_planar.detach()
        losses["Plane_normal_loss"] = torch.mean(similarity_planar)

        # distance_pred_planes = planes_ext * (gt_points - pred_points)

        # distance_pred_planes = distance_pred_planes * torch.reshape(avg_pred_normal, (b, p, 3, 1))
        # distance_pred_planes = torch.sum(distance_pred_planes, dim=2)
        # print(distance_pred_planes)
        # planeMean = createValidMean(plane_mask)
        # distance_loss_plane = l2Loss(-distance_pred_planes,
        #                              torch.zeros_like(distance_pred_planes), planeMean)
        l, h = self.normal_loss(predictions, data)
        losses = mergeInDict(losses, l)
        histograms = mergeInDict(histograms, h)

        return losses, histograms


class PlaneParamsLoss(nn.Module):
    def __init__(self, width=256, height=512):
        super(PlaneParamsLoss, self).__init__()
        self.to3d = Depth2Points(width, height)
        self.sobel = Sobel()

    def forward(self, predictions, data):
        mask = data.get(DataType.Mask, scale=1)[0]
        planes = data.get(DataType.Planes, scale=1)[0] * mask
        # TODO: remove assumptions that one pixel can have only one plane
        plane_mask = torch.sign(torch.sum(planes, dim=1))
        gt_depth = data.get(DataType.Depth, scale=1)[0]
        pred_depth = predictions.get(DataType.Depth, scale=1)[0]
        gt_normals = data.get(DataType.Normals)[0]

        pred_planes_params = predictions.get(DataType.PlaneParams)[0]
        gt_planes_params = toPlaneParams(gt_normals, gt_depth)

        validMean = createValidMean(data.get(DataType.Mask, scale=2)[0])
        diff_loss2x = l1Loss(predictions.get(DataType.Depth, scale=2)[0],
                             data.get(DataType.Depth, scale=2)[0], validMean)
        # print(pred_planes_params.size(), gt_planes_params.size())
        validMean = createValidMean(plane_mask)

        # diff_loss1x = l1Loss(pred_depth, gt_depth, validMean)
        param_loss = l1Loss(pred_planes_params, gt_planes_params, mean_fce=validMean, dim=1)
        similarity = cosineLoss(pred_planes_params, gt_planes_params, dim=1, mean_fce=validMean)

        b, p, h, w = planes.size()
        with torch.no_grad():
            planes_ext = torch.reshape(planes, (b, p, 1, -1))

        pred_points = self.to3d(pred_depth)
        pred_points = torch.reshape(pred_points, (b, 1, 3, -1))

        pred_planes_params = torch.reshape(pred_planes_params, (b, 1, 3, -1))
        pred_planes_params = planeAwarePooling(pred_planes_params, planes_ext)
        # print(pred_points.size(), pred_planes_params.size())
        q_factor = torch.sum(pred_points * pred_planes_params, dim=2, keepdim=True)
        # L1 loss
        q_loss = torch.sum(planes_ext * torch.abs(q_factor - torch.ones_like(q_factor)))
        # TODO: same assumtion about one pixel per plane
        q_loss /= torch.sum(plane_mask)
        return {
            "Param_Loss": param_loss,
            "Param_Similarity_Loss": similarity,
            "Q_Loss": q_loss,
            "Diff_2x_Loss": diff_loss2x,
            # "Diff_1x_Loss": diff_loss1x,
        }
