import torch
import torch.nn as nn

import visdom
from trainers import MonoTrainer
import dataset
from util import set_caffe_param_mult
from run_utils import defineModelParameters, parseArgs, setupGPUDevices

import os.path as osp
import os
import json
from test import test


args = parseArgs(test=False)
checkpoint_dir = osp.join('./experiments/', args.experiment_name)
os.makedirs(checkpoint_dir, exist_ok=True)

with open(osp.join(checkpoint_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))

validation_freq = 1
visualization_freq = 5
validation_sample_freq = -1

model, criterion, parser, image_transformer, depth_transformer = defineModelParameters(
    args.network_type, args.loss_type, args.add_points)

network, criterion, device = setupGPUDevices(
    gpus_list=args.gpu_ids, model=model, criterion=criterion)

vis = visdom.Visdom()
env = args.experiment_name


train_dataloader = torch.utils.data.DataLoader(
    dataset=dataset.OmniDepthDataset(
        root_path=args.dataset_dir,
        path_to_img_list=args.train_list,
        use_sparse_pts=args.add_points,
        transformer_depth=depth_transformer,
        transformer_rgb=image_transformer),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    drop_last=True)

val_dataloader = torch.utils.data.DataLoader(
    dataset=dataset.OmniDepthDataset(
        root_path=args.dataset_dir,
        path_to_img_list=args.val_list,
        use_sparse_pts=args.add_points,
        transformer_depth=depth_transformer,
        transformer_rgb=image_transformer),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    drop_last=True)


# Set up network parameters with Caffe-like LR multipliers
param_list = set_caffe_param_mult(network, args.lr, 0)
optimizer = torch.optim.Adam(params=param_list, lr=args.lr)

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
test()
