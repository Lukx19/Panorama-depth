
import os.path as osp
import os
import argparse
import glob
from enum import Enum
import math
import numpy as np
from PIL import Image
from util import read_tiff
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pickle as pl
from scipy import stats
from matplotlib.ticker import PercentFormatter


def depthBasedHistogram(filename, gt_depths, diffrence, ylabel="Mean error",
                        max_y=1.3, title="Title", log_scale=False):
    fig, ax = plt.subplots()
    bins = 100
    bin_means, bin_edges, binnumber = stats.binned_statistic(gt_depths, diffrence,
                                                             range=(0.0, 8.0), statistic='mean',
                                                             bins=bins)
    bin_stds, bin_edges, binnumber = stats.binned_statistic(gt_depths, diffrence,
                                                            range=(0.0, 8.0), statistic='std',
                                                            bins=bins)

    bin_count, _, _ = stats.binned_statistic(gt_depths, diffrence,
                                             range=(0.0, 8.0), statistic='count',
                                             bins=bins)
    # total_pixels = len(gt_depths)
    # bin_weight = bin_count / total_pixels
    # bin_means /= bin_weight

    bin_labels = [bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2
                  for i in range(len(bin_edges) - 1)]

    ax.errorbar(bin_labels, bin_means, yerr=bin_stds, ecolor='deepskyblue', color='indigo',
                elinewidth=2, linewidth=4, label='both limits(default)')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Distance from camera [m]')
    ax.set_xticks(np.linspace(0.0, 8.0, 10))
    if log_scale:
        ax.set_yscale('log')
    ax.set_ylim((0, max_y))
    ax.set_xlim((1, 8.0))
    # ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.yaxis.grid(True)
    # plt.figure()
    fig.tight_layout()
    fig.savefig(filename)
    fig.show()
    # with open(filename.replace('.png', '.pickle'), 'w') as f:
    pickle_filename = filename.replace('.png', '.pickle')
    pl.dump((fig, ax), open(pickle_filename, 'wb'))


def saveBinnedHistogram(filename, x, vals,
                        xlabel='Distance from camera [m]', ylabel="Mean error",
                        title="Title", log_scale=False, x_range=(0.0, 8.0), bins=100):
    fig, ax = plt.subplots()
    bin_means, bin_edges, binnumber = stats.binned_statistic(x, vals,
                                                             range=x_range, statistic='count',
                                                             bins=bins)

    bin_labels = [bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2
                  for i in range(len(bin_edges) - 1)]
    # print(bin_means)
    ax.bar(bin_labels, bin_means, color='deepskyblue', width=0.2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(np.linspace(x_range[0], x_range[1], 10))

    # ax.set_ylim((0, max_y))
    ax.set_xlim(x_range)
    # ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.yaxis.grid(True)
    # plt.figure()
    fig.tight_layout()
    fig.savefig(filename)
    fig.show()
    # with open(filename.replace('.png', '.pickle'), 'w') as f:
    pickle_filename = filename.replace('.png', '.pickle')
    pl.dump((fig, ax), open(pickle_filename, 'wb'))


def saveHistogram(filename, vals,
                  xlabel=None, ylabel=None,
                  title=None, x_range=(0.0, 8.0), bins=50):
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(vals, bins=bins, range=x_range,
                               weights=np.ones(len(vals)) / len(vals),
                               density=False, facecolor='deepskyblue', alpha=0.75)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xticks(np.linspace(x_range[0], x_range[1], 10))

    # ax.set_ylim((0, max_y))
    ax.set_xlim(x_range)
    # ax.set_xticklabels(labels)
    if title is not None:
        ax.set_title(title)
    ax.yaxis.grid(True)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    fig.tight_layout()
    fig.savefig(filename)

    pickle_filename = filename.replace('.png', '.pickle')
    pl.dump((fig, ax), open(pickle_filename, 'wb'))


class Axis(Enum):
    X = 1
    Y = 2
    Z = 3
    TILT = 4


def getRotationMatrix(axis, theta):
    if axis == Axis.X:
        return np.array([[1, 0, 0, 0, ]
                         [0, math.cos(theta), -math.sin(theta), 0],
                         [0, math.sin(theta), math.cos(theta), 0],
                         [0, 0, 0, 1]])

    if axis == Axis.Y:
        return np.array([[math.cos(theta), 0, math.sin(theta), 0],
                         [0, 1, 0, 0],
                         [-math.sin(theta), 0, math.cos(theta), 0],
                         [0, 0, 0, 1]])

    if axis == Axis.Z:
        return np.array([[math.cos(theta), -math.sin(theta), 0, 0],
                         [math.sin(theta), math.cos(theta), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    if axis == Axis.TILT:
        return np.array([[1, 0, 0, 0],
                         [0, math.cos(theta), -math.sin(theta), 0],
                         [0, math.sin(theta), math.cos(theta), 0],
                         [0, 0, 0, 1]])
        # R = R(1: 3, 1: 3)
    return np.eye(4, 4)


def savePcl(points, rgb, file, normals=None, filter_invalid=False):
    encoded = []
    pt_count = points.shape[0]

    if rgb is None or len(rgb) == 0:
        rgb = np.ones((pt_count, 3)) * 250

    if normals is None or len(normals) == 0:
        normals = np.zeros((pt_count, 3))

    for pt, normal, color in zip(points, normals, rgb):
        if filter_invalid:
            if np.sum(np.abs(color)) == 0 or np.sum(np.abs(pt)) < 0.001:
                continue
        encoded.append("%f %f %f %f %f %f %d %d %d 0\n" %
                       (pt[0], pt[1], pt[2], normal[0], normal[1], normal[2],
                        color[0], color[1], color[2]))

    with open(file, "w") as file:
        file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property float nx
            property float ny
            property float nz
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(encoded), "".join(encoded)))


def createPcl(rgb, depth, cx, cy, fx, fy, scale=1):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb -- color image tensor C x W X H
    depth -- depth image tensor C x W X H
    ply_file -- filename of ply file
    """

    points = []
    colors = []
    w, h, ch = rgb.shape
    for v in range(h):
        for u in range(w):
            color = rgb[u, v]
            Z = depth[u, v] * scale
            if Z == 0:
                continue
            X = (float(u) - cx) * Z / fx
            Y = (float(v) - cy) * Z / fy
            points.append([X, Y, Z])
            colors.append(color)
    return points, colors


def panoDepthToBoxPcl(depth, rgb, box_trans=False):

    Rref = np.array([[0, 0, 1], [0, - 1, 0], [1, 0, 0]])
    camera_center = np.array([0, 0, 0])
    w, h = depth.shape
    f_vir = 80
    cx = w / 2
    cy = h / 2
    fx = f_vir
    fy = f_vir

    gap = int(np.floor(w / 4))
    virtual_cams = []
    points = None
    colors = None
    for i in range(4):
        theta = -(i) * np.pi / 2
        rot = getRotationMatrix(Axis.Y, theta)
        extrinsics = rot[0:3, 0:3] @ Rref
        virtual_cams.append(extrinsics)
        local_depth = depth[:, (gap * i):(gap * (i + 1))]
        local_rgb = rgb[:, (gap * i):(gap * (i + 1)), :]
        loc_points, loc_colors = createPcl(
            local_rgb, local_depth, cx, cy, fx, fy)
        if box_trans:
            loc_points = loc_points @ extrinsics + camera_center
        else:
            loc_points = loc_points + np.array([i * gap + 50, 0, 0])

        if points is None:
            points = loc_points
            colors = loc_colors
        else:
            np.concatenate((points, loc_points), axis=0)
            np.concatenate((colors, loc_colors), axis=0)
        print(loc_points)
    print(points)
    # pcd = open3d.PointCloud()
    # pcd.points = open3d.Vector3dVector(points)
    # pcd.colors = open3d.Vector3dVector(colors)
    return [points, colors]


def panoDepthToPcl(depth, rgb, normals=None, scale=1, mask=None):
    points = []
    colors = []
    n = []
    height, width = depth.shape

    # Camera rotation angles
    hcam_deg = 360
    vcam_deg = 180
    # Camera rotation angles in radians
    hcam_rad = hcam_deg / 180.0 * np.pi
    vcam_rad = vcam_deg / 180.0 * np.pi
    # print(hcam_deg, vcam_deg)
    for v in range(height):
        for u in range(width):
            if mask is not None and mask[v, u] < 0.001:
                continue
            p_theta = (u - width / 2.0) / width * hcam_rad
            p_phi = -(v - height / 2.0) / height * vcam_rad

            # Transform into cartesian coordinates
            radius = (depth[v, u] * scale)
            if radius < 0.001 or radius > 8:
                continue
            # radius = 1
            X = radius * math.cos(p_phi) * math.cos(p_theta)
            Y = radius * math.cos(p_phi) * math.sin(p_theta)
            Z = radius * math.sin(p_phi)

            points.append([X, Y, Z])
            if rgb is not None:
                colors.append(rgb[v, u])
            else:
                colors.append([250, 250, 250])

            if normals is not None:
                n.append(normals[v, u])
            else:
                n.append([0, 0, 0])
    return [np.array(points), np.array(colors), np.array(n)]


def saveDepthMap(depth, filename, max_val=8):
    fig = plt.figure()
    # fig.subplots_adjust(hspace=0.1, wspace=0.01)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.axis('off')
    ax1.imshow(depth, vmin=0, vmax=max_val)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def saveDepthMaps(rgb, gt, pred, filename):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.1, wspace=0.01)
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.axis('off')
    ax1.imshow(rgb)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.axis('off')
    ax2.imshow(gt)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.axis('off')
    ax3.imshow(pred)

    fig.savefig(filename, dpi=300)
    plt.close(fig)


def convert(args):
    output_dir, data = args
    img_path, gt_path, pred_path = data

    basename = osp.basename(img_path)
    basename = osp.splitext(basename)[0]
    # print(basename)
    depth_gt = np.squeeze(read_tiff(gt_path))
    depth_pred = np.squeeze(read_tiff(pred_path))
    rgb = np.array(Image.open(img_path))

    filename = osp.join(output_dir, basename + ".png")
    saveDepthMaps(rgb, depth_gt, depth_pred, filename)

    pcd = panoDepthToPcl(depth_pred, rgb)
    filename = osp.join(output_dir, basename + "_pred.ply")
    savePcl(pcd[0], pcd[1], filename, filter_invalid=True)

    pcd = panoDepthToPcl(depth_gt, rgb)
    filename = osp.join(output_dir, basename + "_gt.ply")
    savePcl(pcd[0], pcd[1], filename)


def visualizePclDepth(results_dir):
    output_dir = osp.join(results_dir, "./visualizations")
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(glob.glob(results_dir + "/*_color.jpg"))
    gt_depths = sorted(glob.glob(results_dir + "/*_gt_depth.tiff"))
    pred_depths = sorted(glob.glob(results_dir + "/*_pred_depth.tiff"))
    if len(gt_depths) == 0:
        gt_depths = pred_depths
    # print(results_dir+"/*_color.jpg")
    # print(images, gt_depths, pred_depths)
    input_args = zip(images, gt_depths, pred_depths)
    input_args = [(output_dir, data) for data in input_args]
    pool = Pool(os.cpu_count() // 2)
    for _ in tqdm(pool.imap_unordered(convert, input_args), total=len(input_args)):
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Generates test, train and validation splits for 360D')

    parser.add_argument('results_folder', type=str, default="../datasets/",
                        help='Dataset storage folder')

    args = parser.parse_args()
    visualizePclDepth(args.results_folder)


def main2():
    parser = argparse.ArgumentParser(
        description='Generates test, train and validation splits for 360D')

    parser.add_argument('results_folder', type=str, default="../datasets/",
                        help='Dataset storage folder')
    parser.add_argument('--regex', type=str, default="/**/*_depth_*.tiff")

    args = parser.parse_args()
    results_dir = args.results_folder

    depths = sorted(glob.glob(results_dir + "/**/" + args.regex, recursive=True))
    # print(results_dir+"/*_color.jpg")
    # print(images, gt_depths, pred_depths)

    for depth_path in depths:
        depth = np.squeeze(read_tiff(depth_path))
        filename = depth_path[:-4] + "ply"

        print(filename)
        pcd = panoDepthToPcl(depth, None)
        savePcl(pcd[0], pcd[1], filename)


if __name__ == "__main__":
    main()
