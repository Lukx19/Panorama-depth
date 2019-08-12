import torch


from trainers import MonoTrainer, save_saples_for_pcl
from dataset import OmniDepthDataset

import json
import os

import os.path as osp
from run_utils import parseArgs, setupPipeline, setupGPUDevices
import subprocess
import numpy as np
import pickle as pl
from visualizations import depthBasedHistogram
# --------------
# PARAMETERS
# --------------


def create_file_list(file_names, test_list, n_files=50, the_best=True):
    file_names.sort(key=lambda v: v[0], reverse=the_best)

    max_count = min(len(file_names), n_files)
    lines = []
    count = 0
    for d1, filename, mean_angle in file_names:
        # print(filename)
        if ("color_0" in filename and "Left" not in filename) or "_0.0" in filename:
            # print("filename is ok")
            cmd = f'grep /{filename} "{test_list}"'
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            (output, err) = p.communicate()
            p.wait()
            lines.append(str(output, 'utf-8'))
            count += 1
        if count > max_count:
            break
    return lines


def test(args=None, best_model=True):
    if args is None:
        args = parseArgs(test=True)
    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint

    expr_name = args.experiment_name

    if osp.exists(expr_name):
        experiment_folder = expr_name
        expr_name = osp.basename(expr_name)
    else:
        experiment_folder = osp.join('./experiments/', expr_name)

    if args.checkpoint is None:
        if best_model:
            checkpoint = osp.join(experiment_folder, "model_best.pth")
        else:
            checkpoint = osp.join(experiment_folder, "checkpoint_latest.pth")

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

    report, (files_scores, gt_depths, depth_diffs, angle_diffs) = trainer.validate(
        checkpoint_path=checkpoint, save_all_predictions=args.save_results, extra_stats=args.save_results)
    with open(osp.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(json.dumps(report, indent=2))

    lines = create_file_list(files_scores, args.test_list, n_files=50, the_best=False)
    with open(f"{results_dir}/list_best_False.txt", "w") as f:
        f.write("".join(lines))

    lines = create_file_list(files_scores, args.test_list, n_files=50, the_best=True)
    with open(f"{results_dir}/list_best_True.txt", "w") as f:
        f.write("".join(lines))

    # files_scores.sort(key=lambda v: v[2], reverse=False)
    with open(f"{results_dir}/file_scores.txt", "w") as f:
        json.dump(files_scores, f)

    if args.save_results:
        pl.dump((gt_depths, depth_diffs, angle_diffs),
                open(osp.join(results_dir, 'histogram_data.pickle'), "wb"))

        abs_depth_difs = np.abs(depth_diffs) / gt_depths
        depthBasedHistogram(osp.join(results_dir, 'depthabs_diff_hist.png'), gt_depths,
                            abs_depth_difs, ylabel="Absolute Relative Error [m]",
                            max_y=0.6, title=expr_name)

        mse_depth_difs = np.abs(depth_diffs)**2
        depthBasedHistogram(osp.join(results_dir, 'depth_mse_hist.png'), gt_depths, mse_depth_difs,
                            ylabel="Mean Square Error [m]", max_y=5, title=expr_name,
                            log_scale=True)

        # depthCountHistogram(osp.join(results_dir, 'depth_hist.png'), gt_depths, np.abs(depth_diffs),
        #                     ylabel="Count Elements", title=expr_name)
        if len(angle_diffs) == len(gt_depths):
            angle_diffs = np.where(angle_diffs > 1, 1, angle_diffs)
            angle_diffs = np.where(angle_diffs < -1, -1, angle_diffs)
            angle_diffs = np.abs(np.arccos(angle_diffs)) * (180 / np.pi)
            depthBasedHistogram(osp.join(results_dir, 'angle_hist.png'), gt_depths, angle_diffs,
                                ylabel="Mean Angle Difference [deg]", max_y=90, title=expr_name)
            # depthCountHistogram(osp.join(results_dir, 'angle_count_hist.png'), gt_depths,
            # angle_diffs, ylabel="Count Elements", title=expr_name)

    return report


if __name__ == "__main__":
    test()
