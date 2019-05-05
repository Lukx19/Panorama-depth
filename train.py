import torch

import visdom
from trainers import MonoTrainer
import dataset
from util import set_caffe_param_mult, load_encoder_weights
from run_utils import setupPipeline, parseArgs, setupGPUDevices

import os.path as osp
import os
import json


args = parseArgs(test=False)
torch.manual_seed(19)
experiment_name = args.network_type + "_" + args.loss_type + "_" + args.experiment_name
checkpoint_dir = osp.join('./experiments/', experiment_name)
os.makedirs(checkpoint_dir, exist_ok=True)

with open(osp.join(checkpoint_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))

validation_freq = 1
visualization_freq = 100
validation_sample_freq = -1

model, (criterion, loss_sum_fce), parser, image_transformer, depth_transformer = setupPipeline(
    args.network_type, args.loss_type, args.add_points)

network, criterion, device = setupGPUDevices(
    gpus_list=args.gpu_ids, model=model, criterion=criterion)

vis = visdom.Visdom()
env = experiment_name


train_dataloader = torch.utils.data.DataLoader(
    dataset=dataset.OmniDepthDataset(
        root_path=args.dataset_dir,
        path_to_img_list=args.train_list,
        use_sparse_pts=args.add_points,
        transformer_depth=depth_transformer,
        transformer_rgb=image_transformer,
        use_normals=args.load_normals,
        use_planes=args.load_planes,
    ),
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=args.workers,
    drop_last=True)

val_dataloader = torch.utils.data.DataLoader(
    dataset=dataset.OmniDepthDataset(
        root_path=args.dataset_dir,
        path_to_img_list=args.val_list,
        use_sparse_pts=args.add_points,
        transformer_depth=depth_transformer,
        transformer_rgb=image_transformer,
        use_normals=args.load_normals,
        use_planes=args.load_planes,
    ),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=args.workers,
    drop_last=False)


# Set up network parameters with Caffe-like LR multipliers
param_list = set_caffe_param_mult(network, args.lr, 0)
optimizer = torch.optim.Adam(params=param_list, lr=args.lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=args.decay_step_size,
                                            gamma=args.decay_lr)


trainer = MonoTrainer(
    experiment_name,
    network,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    checkpoint_dir,
    device,
    parse_fce=parser,
    sum_losses_fce=loss_sum_fce,
    visdom=[vis, env],
    scheduler=scheduler,
    num_epochs=args.epochs,
    validation_freq=validation_freq,
    visualization_freq=visualization_freq,
    validation_sample_freq=validation_sample_freq)

# trainer.setDryRun(True)
if args.encoder_weights is not None:
    load_encoder_weights(network, args.encoder_weights)
# trainer.visualizeNetwork(args.checkpoint)
trainer.train(args.checkpoint, args.only_weights)
# test(experiment_name)
