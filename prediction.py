import torch


from trainers import MonoTrainer, save_saples_for_pcl
from dataset import OmniDepthDataset, ImageDataset, prediction_rgb_trasformer

import json
import os

import os.path as osp
from run_utils import parseArgs, setupPipeline, setupGPUDevices
# --------------
# PARAMETERS
# --------------


def predict(experiment_name=None):
    args = parseArgs(predict=True)
    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint
    if experiment_name is None:
        expr_name = args.experiment_name
    else:
        expr_name = experiment_name

    if osp.exists(expr_name):
        experiment_folder = expr_name
        expr_name = osp.basename(expr_name)
    else:
        experiment_folder = osp.join('./experiments/', expr_name)

    if args.checkpoint is None:
        checkpoint = osp.join(experiment_folder, "model_best.pth")

    print("Using checkpoint ", checkpoint)

    results_dir = osp.join(experiment_folder, "./predictions/")
    os.makedirs(results_dir, exist_ok=True)

    with open(osp.join(results_dir, 'predict_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(json.dumps(args.__dict__, indent=2))

    validation_freq = 1
    visualization_freq = 5
    validation_sample_freq = -1

    model, _, parser, image_transformer, depth_transformer = setupPipeline(
        network_type=args.network_type, loss_type=None, add_points=args.add_points,
        empty_points=args.empty_points, args=args)

    network, _, device = setupGPUDevices(
        gpus_list=args.gpu_ids, model=model, criterion=None)

    test_dataloader = torch.utils.data.DataLoader(
        dataset=ImageDataset(
            image_folder=args.image_folder,
            transformer_rgb=prediction_rgb_trasformer
        ),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False)

    trainer = MonoTrainer(
        expr_name,
        network,
        None,
        test_dataloader,
        None,
        None,
        results_dir,
        device,
        parse_fce=parser,
        save_samples_fce=save_saples_for_pcl,
        visdom=None,
        scheduler=None,
        num_epochs=None,
        validation_freq=validation_freq,
        visualization_freq=visualization_freq,
        validation_sample_freq=validation_sample_freq)

    trainer.predict(checkpoint_path=checkpoint)


if __name__ == "__main__":
    predict()
