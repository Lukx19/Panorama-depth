import torch
import torch.nn as nn

import visdom

from trainers import MonoTrainer, parse_data, parse_data_sparse_depth
from network import *
from criteria import *
from dataset import *
from util import mkdirs, set_caffe_param_mult
import argparse
import os.path as osp
import os
import json

# --------------
# PARAMETERS
# --------------
parser = argparse.ArgumentParser(
    description='Generates test, train and validation splits for 360D')
parser.add_argument('experiment_name', type=str,
                    help='Name of this experiment. Used to creat folder in checkpoint folder.')

parser.add_argument('--network_type', default="RectNet", type=str,
                    help='UResNet or RectNet or RectNet2')

parser.add_argument('--loss_type', default="Revis", type=str,
                    help='MultiScale or Revis')

parser.add_argument('--add_points', action="store_true", default=False,
                    help='In addition to monocular image also add sparse points to training.')

parser.add_argument('--cspn', action="store_true", default=False,
                    help='Activates recursive refinement CSPN layer.')

parser.add_argument('--dataset_dir', type=str, default="../datasets/",
                    help='Dataset storage folder')

parser.add_argument('--train_list', type=str,
                    default="./data_splits/original_p100_d20_train_split.txt",
                    help='Trainig list with data filenames used for training')

parser.add_argument('--val_list', type=str,
                    default="./data_splits/original_p100_d20_test_split.txt",
                    help='Validation list with data samples used in model validation')

parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to checkpoint')

parser.add_argument('--only_weights', action="store_true", default=False,
                    help='Use only weights from checkpoint. Optimizer state or epoch information is not restored.')

parser.add_argument('--gpu_ids', default='0,1', type=str,
                    help='Ids of GPU to use for training')

parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--decay_step_size', default=3, type=int)
parser.add_argument('--decay_lr', default=0.5, type=float)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=2e-4, type=float, help='Learning rate')
args = parser.parse_args()


checkpoint_dir = osp.join('./experiments/', args.experiment_name)
os.makedirs(checkpoint_dir, exist_ok=True)

with open(osp.join(checkpoint_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))

validation_freq = 1
visualization_freq = 5
validation_sample_freq = -1

device_ids = [int(s) for s in args.gpu_ids.split(',')]
print(device_ids)

# -------------------------------------------------------
# Fill in the rest
vis = visdom.Visdom()
env = args.experiment_name
device = torch.device('cuda', device_ids[0])

in_channels = 3
parser = parse_data
if args.add_points:
    in_channels = 4
    parser = parse_data_sparse_depth

# UResNet
if args.network_type == 'UResNet':
    model = UResNet(in_channels)
    alpha_list = [0.445, 0.275, 0.13]
    beta_list = [0.15, 0., 0.]
# RectNet
elif args.network_type == 'RectNet':
    model = RectNet(in_channels, cspn=args.cspn)
    alpha_list = [0.535, 0.272]
    beta_list = [0.134, 0.068, ]
elif args.network_type == 'RectNet2':
    model = RectNet2(in_channels, cspn=args.cspn)
    alpha_list = [0.535, 0.272]
    beta_list = [0.134, 0.068, ]
else:
    assert False, 'Unsupported network type'

if args.loss_type == "MultiScale":
    criterion = MultiScaleL2Loss(alpha_list, beta_list).to(device)
elif args.loss_type == "Revis":
    criterion = GradLoss().to(device)
else:
    assert False, 'Unsupported loss type'

# -------------------------------------------------------
# Set up the training routine
if len(device_ids) > 1:
    network = nn.DataParallel(
        model.float(),
        device_ids=device_ids).to(device)
elif len(device_ids) == 1:
    network = model.float().to(device)
else:
    assert False, 'Cannot run without specifying GPU ids'

train_dataloader = torch.utils.data.DataLoader(
    dataset=OmniDepthDataset(
        root_path=args.dataset_dir,
        path_to_img_list=args.train_list,
        use_sparse_pts=args.add_points),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    drop_last=True)

val_dataloader = torch.utils.data.DataLoader(
    dataset=OmniDepthDataset(
        root_path=args.dataset_dir,
        path_to_img_list=args.val_list,
        use_sparse_pts=args.add_points),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    drop_last=True)


# Set up network parameters with Caffe-like LR multipliers
param_list = set_caffe_param_mult(network, args.lr, 0)
optimizer = torch.optim.Adam(
    params=param_list,
    lr=args.lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=args.decay_step_size,
                                            gamma=args.decay_lr)


trainer = MonoTrainer(
    args.experiment_name,
    network,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    checkpoint_dir,
    device,
    parse_fce=parser,
    visdom=[vis, env],
    scheduler=scheduler,
    num_epochs=args.epochs,
    validation_freq=validation_freq,
    visualization_freq=visualization_freq,
    validation_sample_freq=validation_sample_freq)


trainer.train(args.checkpoint, args.only_weights)
