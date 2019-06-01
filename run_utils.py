import torch
import torch.nn as nn
from network import UResNet, RectNet, RectNetCSPN, DoubleBranchNet
from criteria import (GradLoss, MultiScaleL2Loss, NormSegLoss,
                      PlaneNormSegLoss, PlaneParamsLoss, CosineNormalsLoss,
                      SphericalNormalsLoss)
import argparse
import dataset
from trainers import parse_data, genSparsePontsParser, genTotalLoss
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def setupPipeline(network_type, loss_type, add_points, empty_points, loss_scales={}):
    in_channels = 3
    parser = parse_data
    if add_points:
        in_channels = 4
        parser = genSparsePontsParser(empty_points)

    rgb_transformer = dataset.default_transformer
    depth_transformer = dataset.default_depth_transformer

    # UResNet
    if network_type == 'UResNet':
        model = UResNet(in_channels)
        alpha_list = [0.445, 0.275, 0.13]
        beta_list = [0.15, 0., 0.]

    elif network_type == 'RectNet':
        model = RectNet(in_channels, cspn=False)
        alpha_list = [0.535, 0.272]
        beta_list = [0.134, 0.068, ]
    elif network_type == 'RectNetSegNormals':
        model = RectNet(in_channels, cspn=False, normal_est=True, segmentation_est=True)

    elif network_type == 'RectNetSphereNormals':
        model = RectNet(in_channels, cspn=False, normal_est=True,
                        segmentation_est=True, normals_est_type="sphere")
    elif network_type == 'RectNetPlanes':
        torch.autograd.set_detect_anomaly(True)
        model = RectNet(in_channels, cspn=False, normal_est=True,
                        segmentation_est=True, calc_planes=True,
                        normals_est_type="standart")
    elif network_type == 'RectNetPlanesSphere':
        torch.autograd.set_detect_anomaly(True)
        model = RectNet(in_channels, cspn=False, normal_est=True,
                        segmentation_est=True, calc_planes=True,
                        normals_est_type="sphere")
    elif network_type == 'RectNetPad':
        model = RectNet(in_channels, cspn=False, reflection_pad=True)
        alpha_list = [0.535, 0.272]
        beta_list = [0.134, 0.068, ]
    elif network_type == 'RectNetCSPN':
        model = RectNetCSPN(in_channels, cspn=True)
        alpha_list = [0.535, 0.272]
        beta_list = [0.134, 0.068, ]
    elif network_type == "DBNet":
        model = DoubleBranchNet(in_channels)
    else:
        assert False, 'Unsupported network type'

    criterion = None
    if loss_type is not None:
        if loss_type == "MultiScale":
            criterion = MultiScaleL2Loss(alpha_list, beta_list)
        elif loss_type == "Revis":
            criterion = GradLoss()
        elif loss_type == "Revis_all":
            criterion = GradLoss(all_levels=True)
        elif loss_type == "Revis_normals":
            criterion = GradLoss(all_levels=True, depth_normals=True)
        elif loss_type == "Revis_adaptive":
            criterion = GradLoss(all_levels=True, adaptive=True)
        elif loss_type == "Revis_Normal_Seg":
            criterion = NormSegLoss()
        elif loss_type == "PlaneNormSegLoss":
            normal_loss = CosineNormalsLoss()
            criterion = PlaneNormSegLoss(normal_loss=normal_loss)
            loss_scales["Plane_dist_plane_loss_Ad"] = 0.0
            # loss_scales["Plane_dist_plane_loss"] = 10.0
            # loss_scales["revis_l1_dist_1"] = 10.0
            # loss_scales["revis_l1_dist_2"] = 10.0
            # loss_scales["Pixel_normal_similarity_loss"] = 0.0
            # loss_scales["Plane_normal_similarity_loss"] = 0.0
            # loss_scales["Segmentation_Loss"] = 0.0
        elif loss_type == "PlaneParamsLoss":
            loss_scales["Plane_dist_plane_loss_Ad"] = 0.0
            criterion = PlaneParamsLoss()
        elif loss_type == "PlaneNormClassSegLoss":
            loss_scales["Plane_dist_plane_loss_Ad"] = 0.0
            normal_loss = SphericalNormalsLoss()
            criterion = PlaneNormSegLoss(normal_loss=normal_loss)
            # loss_scales["Plane_dist_plane_loss"] = 10.0
            # loss_scales["revis_l1_dist_1"] = 10.0
            # loss_scales["revis_l1_dist_2"] = 10.0
        else:
            assert False, 'Unsupported loss type'
    return model, (criterion, genTotalLoss(loss_scales)), parser, rgb_transformer, depth_transformer


def parseLossScales(loss_str: str):
    loss_scales = {}
    if loss_str == "":
        return loss_scales
    losses = loss_str.split(",")
    for loss in losses:
        loss_name, loss_value = tuple(loss.split(":"))
        loss_scales[loss_name.strip()] = float(loss_value.strip())
    return loss_scales


def parseArgs(test=False, predict=False):
    description = 'Training script for Panodepth training procedure'
    if test:
        description = 'Testing script for Panodepth'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('experiment_name', type=str,
                        help='Name of this experiment. Used to creat folder in checkpoint folder.')

    parser.add_argument('--network_type', default="RectNet", type=str,
                        help="UResNet or RectNet or RectNetCSPN \
                            or UResNet_Resnet or RectNetPad or DBNet or RectNetPlaneParams")

    parser.add_argument('--add_points', action="store_true", default=False,
                        help='In addition to monocular image also add sparse points to training.')

    parser.add_argument('--empty_points', action="store_true", default=False,
                        help='Make channel with sparse points empty')

    parser.add_argument('--load_normals', action="store_true", default=False,
                        help='Load normals from dataset')

    parser.add_argument('--load_planes', action="store_true", default=False,
                        help='Load planes from dataset')

    parser.add_argument('--dataset_dir', type=str, default="../datasets/Omnidepth/",
                        help='Dataset storage folder')

    parser.add_argument('--gpu_ids', default='0,1', type=str,
                        help='Ids of GPU to use for training')
    parser.add_argument('--workers', default=4, type=int)

    if test:
        parser.add_argument('--test_list', type=str,
                            default="./data_splits/original_p100_d20_test_split.txt",
                            help='Validation list with data samples used in model validation')

        parser.add_argument('--checkpoint', type=str, default=None,
                            help='Path to checkpoint. Default checkpoint is used based \
                            on experiment folder and best model in this folder')

        parser.add_argument('--save_results', action="store_true", default=False,
                            help='Save all generated outputs and inputs')
    elif predict:
        parser.add_argument('--image_folder', type=str,
                            default="./",
                            help='Path to folder with images used for depth prediction')

        parser.add_argument('--checkpoint', type=str, default=None,
                            help='Path to checkpoint. Default checkpoint is used based \
                            on experiment folder and best model in this folder')
    else:
        parser.add_argument('--train_list', type=str,
                            default="./data_splits/original_p100_d20_train_split.txt",
                            help='Trainig list with data filenames used for training')

        parser.add_argument('--val_list', type=str,
                            default="./data_splits/original_p100_d20_test_split.txt",
                            help='Validation list with data samples used in model validation')

        parser.add_argument('--checkpoint', type=str, default=None,
                            help='Path to checkpoint')

        parser.add_argument('--only_weights', action="store_true", default=False,
                            help='Use only weights from checkpoint. Optimizer state or \
                            epoch information is not restored.')

        parser.add_argument('--encoder_weights', type=str, default=None,
                            help='Path to checkpoint with encoder weights')

        parser.add_argument('--loss_type', default="Revis", type=str,
                            help='MultiScale or Revis or PlaneNormSegLoss or PlaneParamsLoss')
        parser.add_argument('--loss_scales', default="", type=str,
                            help=("""Define individual loss function scaling in a form
                                   loss_name:10.0, loss_name2:0.0"""))

        parser.add_argument('--batch_size', default=8,
                            type=int, help='Batch size')
        parser.add_argument('--decay_step_size', default=3, type=int)
        parser.add_argument('--decay_lr', default=0.5, type=float)
        parser.add_argument('--epochs', default=10, type=int)
        parser.add_argument('--lr', default=2e-4,
                            type=float, help='Learning rate')

    return parser.parse_args()


def setupGPUDevices(gpus_list, model, criterion=None):
    if gpus_list == "cpu":
        device = torch.device("cpu")
        network = model.float().to(device)
        return network, criterion, device
    device_ids = [int(s) for s in gpus_list.split(',')]
    print(device_ids)
    print('cuda version=', torch.version.cuda)
    print('cudnn version=', torch.backends.cudnn.version())
    print("cudnn enables=", torch.backends.cudnn.enabled)
    print("cudnn benchmark=", torch.backends.cudnn.benchmark)

    device = torch.device('cuda', device_ids[0])
    torch.cuda.set_device(device)
    if len(device_ids) > 1:
        network = nn.DataParallel(
            model.float(),
            device_ids=device_ids).to(device)
    elif len(device_ids) == 1:
        network = model.float().to(device)
    else:
        assert False, 'Cannot run without specifying GPU ids'

    if criterion is not None:
        criterion = criterion.to(device)

    return network, criterion, device
