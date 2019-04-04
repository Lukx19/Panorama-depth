import torch


from trainers import MonoTrainer, save_saples_for_pcl
from dataset import OmniDepthDataset

import json
import os

import os.path as osp
from run_utils import parseArgs, defineModelParameters, setupGPUDevices
# --------------
# PARAMETERS
# --------------


def test(experiment_name=None):
    args = parseArgs(test=True)
    torch.manual_seed(19)
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

    results_dir = osp.join(experiment_folder, "./results/")
    os.makedirs(results_dir, exist_ok=True)

    with open(osp.join(results_dir, 'test_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(json.dumps(args.__dict__, indent=2))

    validation_freq = 1
    visualization_freq = 5
    validation_sample_freq = -1

    model, _, parser, image_transformer, depth_transformer = defineModelParameters(
        network_type=args.network_type, loss_type=None, add_points=args.add_points)

    network, _, device = setupGPUDevices(
        gpus_list=args.gpu_ids, model=model, criterion=None)

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

    report = trainer.validate(
        checkpoint_path=checkpoint, save_all_predictions=args.save_results)

    with open(osp.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(report)


if __name__ == "__main__":
    test()
