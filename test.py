import torch


from trainers import MonoTrainer, save_saples_for_pcl
from dataset import OmniDepthDataset

import json
import os

import os.path as osp
from run_utils import parseArgs, setupPipeline, setupGPUDevices
import subprocess
# --------------
# PARAMETERS
# --------------


def create_file_list(file_names, test_list, results_dir, n_files=50, the_best=True):
    file_names.sort(key=lambda v: v[0], reverse=the_best)

    max_count = min(len(file_names), n_files)
    lines = []
    for i in range(max_count):
        d1, filename = file_names[i]
        cmd = f'grep {filename} "{test_list}"'
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p.wait()
        lines.append(str(output, 'utf-8'))

    with open(f"{results_dir}/list_best_{the_best}.txt", "w") as f:
        f.write("".join(lines))


def test(args=None):
    if args is None:
        args = parseArgs(test=True)
    torch.manual_seed(19)
    checkpoint = args.checkpoint

    expr_name = args.experiment_name

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

    model, _, parser, image_transformer, depth_transformer = setupPipeline(
        network_type=args.network_type, loss_type=None, add_points=args.add_points,
        empty_points=args.empty_points, args=args)

    network, _, device = setupGPUDevices(
        gpus_list=args.gpu_ids, model=model, criterion=None)

    test_dataloader = torch.utils.data.DataLoader(
        dataset=OmniDepthDataset(
            root_path=args.dataset_dir,
            path_to_img_list=args.test_list,
            use_sparse_pts=args.add_points,
            transformer_depth=depth_transformer,
            transformer_rgb=image_transformer,
            use_normals=args.load_normals,
            use_planes=args.load_planes,
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

    report, files_scores = trainer.validate(
        checkpoint_path=checkpoint, save_all_predictions=args.save_results)

    with open(osp.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(json.dumps(report, indent=2))

    create_file_list(files_scores, args.test_list, results_dir, n_files=50, the_best=False)
    create_file_list(files_scores, args.test_list, results_dir, n_files=50, the_best=True)
    return report


if __name__ == "__main__":
    test()
