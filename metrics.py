import torch
from network import planeAwarePooling
from modules import Depth2Points
import numpy as np
from scipy import ndimage
from skimage import feature
# ==========================
# Depth Prediction Metrics
# ==========================


def abs_rel_error(pred, gt, mask):
    '''Compute absolute relative difference error'''
    return ((pred[mask > 0] - gt[mask > 0]).abs() / gt[mask > 0]).mean()


def sq_rel_error(pred, gt, mask):
    '''Compute squared relative difference error'''
    return (((pred[mask > 0] - gt[mask > 0]) ** 2) / gt[mask > 0]).mean()


def lin_rms_sq_error(pred, gt, mask):
    '''Compute the linear RMS error except the final square-root step'''
    return torch.sqrt(((pred[mask > 0] - gt[mask > 0]) ** 2).mean())


def log_rms_sq_error(pred, gt, mask):
    '''Compute the log RMS error except the final square-root step'''
    # Compute a mask of valid values
    mask = (mask > 0) & (pred > 1e-7) & (gt > 1e-7)
    return torch.sqrt(((pred[mask].log() - gt[mask].log()) ** 2).mean())


def delta_inlier_ratio(pred, gt, mask, degree=1, base=1.25):
    '''Compute the delta inlier rate to a specified degree (def: 1)'''
    return (torch.max(pred[mask > 0] / gt[mask > 0], gt[mask > 0] / pred[mask > 0]) < (base ** degree)).float().mean()


def delta_normal_angle_ratio(pred, gt, mask, degree=11.25):
    val = torch.nn.functional.cosine_similarity(pred, gt, dim=1)
    mask = torch.squeeze(mask)
    val = torch.squeeze(val)
    val = val[mask > 0]
    tresh = np.cos(degree * np.pi / 180)
    # cos(0) = 1 and cos(1) = 0
    return (val > tresh).float().mean()


def normal_stats(pred, gt, mask):
    b, c, h, w = mask.size()
    assert b == 1
    mask = torch.reshape(mask, (1, h, w))
    val = torch.nn.functional.cosine_similarity(pred, gt, dim=1)[mask > 0]

    to_deg = 180 / np.pi
    mean = torch.acos(torch.mean(val))
    median = torch.acos(torch.median(val))
    return mean * to_deg, median * to_deg


def PCA(data, k=3):
    """Calculates PCA on data

    Parameters
    ----------
    data : Tensor BxCxN
        PCA will be caluclated over C and N dimention
    k : int, optional
        number of dimentions to keep after PCA , by default 1

    Returns
    -------
    Tensor BxCxk
        Resulting
    """
    # preprocess the data
    results = []
    for batch_id in range(data.size(0)):
        X = data[batch_id, :, :]
        X_mean = torch.mean(X, dim=0, keepdim=True)
        X = X - X_mean
        # svd
        U, S, V = torch.svd(torch.t(X))
        # print(U.size(), X.size(), V.size())
        # print(S)
        results.append(V[:, k:k + 1])

    return torch.stack(results, dim=0)


def planarity_errors(pred, gt_normals, planes, mask):
    b, p, h, w = planes.size()
    if b > 1:
        raise ValueError("Batch size must be 1 in evaluation")

    to3d = Depth2Points(h, w)
    mask = mask.cpu()
    pred_points = to3d(pred.cpu())
    pred_points = pred_points * mask
    gt_normals = torch.reshape(gt_normals.cpu(), (b, 1, 3, -1))
    planes = torch.reshape(planes.cpu(), (b, p, 1, -1))
    # BxPx3x1 where B=1
    avg_gt_normals = planeAwarePooling(gt_normals, planes)
    avg_gt_normals = torch.reshape(avg_gt_normals, (p, 3, 1))

    pred_points = torch.reshape(pred_points, (1, 3, -1))
    planes = torch.reshape(planes, (p, 1, -1))

    planar_points = planes * pred_points

    # Px3x1
    pred_normals = PCA(planar_points, k=1)
    # print(pred_normals.size(), planar_points.size())
    # Px3x1
    centroids = torch.mean(planar_points, dim=2, keepdim=True)
    # print(centroids.size())
    d = - torch.sum(pred_normals * centroids, dim=1)

    # deviation of fitted 3D plane
    pe_flat = torch.std(torch.sum(planar_points * pred_normals, dim=1) + d) * 100.0

    # flip normals Px1x1
    flip_mask = (torch.sum(pred_normals * avg_gt_normals, dim=1, keepdim=True) < 0).float()
    # flip mask has value -1 for elements which shoudl be flipped and 1 for other
    flip_mask = torch.ones_like(flip_mask) + flip_mask * -2
    pred_normals = pred_normals * flip_mask

    norm = torch.cross(avg_gt_normals, pred_normals)
    norm = torch.norm(norm, p=2, dim=1, keepdim=False) + 1e-4
    angle = torch.sum(avg_gt_normals * pred_normals, dim=1)
    to_deg = 180. / np.pi
    # print(norm.size(), angle.size())
    # PE_ori: 3D angle error between ground truth plane and normal vector of fitted plane
    orient_err = torch.atan2(norm, angle) * to_deg
    orient_err = orient_err.mean()
    # print(orient_err, orient_err.size())
    return pe_flat, orient_err


def distance_to_plane(pred_pts, gt_pts, gt_normals, planes):
    b, p = planes.size()[0:2]
    planes = torch.reshape(planes, (b, p, -1))
    # B x P x 1
    valid_per_plane = torch.sum(planes.detach(), dim=2, keepdim=True)

    # #  avoid division by 0
    valid_per_plane[valid_per_plane < 0.0001] = 1

    distance = (pred_pts - gt_pts) * gt_normals
    distace_to_gt_plane = (torch.sum(distance, dim=1))
    distace_to_gt_plane = torch.abs(distace_to_gt_plane)
    distace_to_gt_plane = torch.reshape(distace_to_gt_plane, (b, 1, -1))
    planar_val = planes * distace_to_gt_plane
    val = planar_val
    val = torch.sqrt(((val ** 2).mean()))
    # val = torch.sum(planar_val, dim=1, keepdim=True) / valid_per_plane
    val = torch.mean(val)
    return val


"""
Original code modified to work with pytorch

Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, Marco Körner:
Evaluation of CNN-based Single-Image Depth Estimation Methods.
European Conference on Computer Vision (ECCV) Workshops, 2018).
"""


def directed_depth_error(pred, gt, mask, thr):
    # exclude masked invalid and missing measurements
    gt = gt[mask > 0]
    pred = pred[mask > 0]

    n_pixels = torch.sum(mask)

    gt[gt <= thr] = 1  # assign depths closer than 'thr' as '1s'
    gt[gt > thr] = 0  # assign depths further than 'thr' as '0s'
    pred[pred <= thr] = 1
    pred[pred > thr] = 0

    diff = pred - gt  # compute difference map

    dde_0 = torch.sum(diff == 0) / n_pixels
    dde_m = torch.sum(diff == 1) / n_pixels
    dde_p = torch.sum(diff == -1) / n_pixels

    return dde_0, dde_m, dde_p


def direct_depth_accuracy(pred, gt, mask, thr):
    val = torch.abs(gt - pred)[mask > 0]
    val = (val < thr).float().mean()
    return val


def depth_boundary_error(pred, gt, mask):
    def normalize(image):
        image_normed = image.copy().astype('f')
        # image_normed[image_normed == 0] = np.nan
        image_normed = image_normed - np.nanmin(image_normed)
        image_normed = image_normed / np.nanmax(image_normed)
        return image_normed

    mask = torch.squeeze(mask).cpu().numpy()
    pred = torch.squeeze(pred).cpu().numpy()
    gt = torch.squeeze(gt).cpu().numpy()

    # normalize est depth map from 0 to 1
    pred_normalized = normalize(pred)
    gt_normalized = normalize(gt)

    # print(gt_normalized.shape, np.any(np.isnan(gt_normalized)), np.any(np.isinf(gt_normalized)))
    # print(pred_normalized.shape, np.any(np.isnan(pred_normalized)), np.any(np.isinf(pred_normalized)))
    # apply canny filter
    edges_est = feature.canny(pred_normalized, sigma=np.sqrt(2), low_threshold=0.1,
                              high_threshold=0.2)
    edges_gt = feature.canny(gt_normalized, sigma=np.sqrt(2), low_threshold=0.1,
                             high_threshold=0.2)

    # remove edges from masked regions
    edges_gt = edges_gt * mask
    edges_est = edges_est * mask
    # plt.imshow(edges_est)

    # compute distance transform for chamfer metric
    D_gt = ndimage.distance_transform_edt(1 - edges_gt)
    D_est = ndimage.distance_transform_edt(1 - edges_est)

    max_dist_thr = 10.  # Threshold for local neighborhood

    mask_D_gt = D_gt < max_dist_thr  # truncate distance transform map

    E_fin_est_filt = edges_est * mask_D_gt  # compute shortest distance for all predicted edges

    # assign MAX value if no edges could be found in prediction
    if np.sum(E_fin_est_filt) == 0:
        dbe_acc = max_dist_thr
        dbe_com = max_dist_thr
    else:
        # accuracy: directed chamfer distance
        dbe_acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(E_fin_est_filt)
        # completeness: directed chamfer distance (reversed)
        dbe_com = np.nansum(D_est * edges_gt) / np.nansum(edges_gt)

    return dbe_acc, dbe_com


# import numpy as np
# import cv2
# imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
# imgrgb = imgrgb.astype(np.float32)
# model = 'structured edge/model.yml'
# retval = cv2.ximgproc.createStructuredEdgeDetection(model)
# out = retval.detectEdges(imgrgb)
