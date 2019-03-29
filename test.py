import torch
import torch.nn as nn
import torchvision.transforms as Tt

from trainers import MonoTrainer, parse_data, parse_data_sparse_depth
from network import UResNet, RectNet
from dataset import OmniDepthDataset
import argparse
import json
import os
from PIL import Image
import os.path as osp
import numpy as np
import cv2
from run_utils import parseArgs, defineModelParameters
# --------------
# PARAMETERS
# --------------


def save_depth(filename, tensor, scale):
    img = tensor.cpu().numpy()
    img = np.squeeze(img)
    # img = np.transpose(img, (1, 2, 0))
    print(img.shape)
    img *= scale
    cv2.imwrite(filename, img)
    # img = Tt.functional.to_pil_image(img)
    # img = Image.fromarray(img, mode="I")
    # print(img)
    # img.save(filename)


def test():
    args = parseArgs(test=True)
    checkpoint = args.checkpoint
    if args.checkpoint is None:
        checkpoint = osp.join(
            './experiments/', args.experiment_name, "model_best.pth")

    print("Using checkpoint ", checkpoint)

    results_dir = osp.join(
        './experiments/', args.experiment_name, "./results/")
    os.makedirs(results_dir, exist_ok=True)

    with open(osp.join(results_dir, 'test_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(json.dumps(args.__dict__, indent=2))

    validation_freq = 1
    visualization_freq = 5
    validation_sample_freq = -1

    device_ids = [int(s) for s in args.gpu_ids.split(',')]
    print(device_ids)

    # -------------------------------------------------------
    # Fill in the rest
    device = torch.device('cuda', device_ids[0])

    model, _, parser, image_transformer, depth_transformer = defineModelParameters(
        network_type=args.network_type, loss_type=None,  add_points=args. add_points)

    # -------------------------------------------------------
    # Set up the test routine
    if len(device_ids) > 1:
        network = nn.DataParallel(
            model.float(),
            device_ids=device_ids).to(device)
    elif len(device_ids) == 1:
        network = model.float().to(device)
    else:
        assert False, 'Cannot run without specifying GPU ids'

    test_dataloader = torch.utils.data.DataLoader(
        dataset=OmniDepthDataset(
            root_path=args.dataset_dir,
            path_to_img_list=args.test_list,
            use_sparse_pts=args.add_points,
            transformer_depth=depth_transformer,
            transformer_rgb=image_transformer),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False)

    trainer = MonoTrainer(
        args.experiment_name,
        network,
        None,
        test_dataloader,
        None,
        None,
        results_dir,
        device,
        parse_fce=parser,
        visdom=None,
        scheduler=None,
        num_epochs=None,
        validation_freq=validation_freq,
        visualization_freq=visualization_freq,
        validation_sample_freq=validation_sample_freq)

    report, results = trainer.evaluate(
        checkpoint_path=checkpoint, only_predict=True)

    with open(osp.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(report)

    for result in results:
        unnorm_depth = 8
        to_mm_depth = 1000
        name = result['data']['name'][0]
        name = osp.join(results_dir, name)
        gt = result['data']['gt'][0]
        sparse_points = result['data']['sparse_depth'][0]
        prediction = result['output'][0]
        img = result['data']['image'][0]

        save_depth(name + "_gt_depth.exr", gt, 1)
        save_depth(name + "_pred_depth.exr", prediction, 1)
        save_depth(name + "_sparse_depth.exr",
                   sparse_points, unnorm_depth)

        color_img = Tt.functional.to_pil_image(img.cpu())
        color_img.save(name+"_color.jpg")

        # result['data']['original_image'].write(
        # osp.join(results_dir, name+"_color.jpg"))


if __name__ == "__main__":
    test()
