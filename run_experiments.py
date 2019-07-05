from tqdm import tqdm
from multiprocessing import Pool
import os
import glob
import os.path as osp
import argparse
import json
import test
from types import SimpleNamespace
from collections import defaultdict
import pandas as pd
import visualizations


def main():
    parser = argparse.ArgumentParser(
        description='Generate new test metrics')
    parser.add_argument('experiments_folder', type=str, help='Path to folder')
    parser.add_argument('--test_list', type=str,
                        default="./data_splits/original_p100_d20_test_split.txt",
                        help='Validation list with data samples used in model validation')

    parser.add_argument('--save_results', action="store_true", default=False,
                        help='Save all generated outputs and inputs')

    parser.add_argument('--run_test', action="store_true", default=False,
                        help='Run network test')

    parser.add_argument('--dataset_dir', type=str, default="../datasets/Omnidepth/",
                        help='Dataset storage folder')

    args = parser.parse_args()

    if (not osp.exists(args.experiments_folder)):
        print("Directory: {} does not exist.".format(args.experiments_folder))
        return

    args_files = sorted(glob.glob(args.experiments_folder + '/**/test_args.txt', recursive=True))
    experiments = []
    for args_file in tqdm(args_files):
        test_results_folder, _ = osp.split(args_file)
        experiment_folder, _ = osp.split(test_results_folder)
        _, experiment_name = osp.split(experiment_folder)
        # print(args_file, test_results_folder, experiment_folder, experiment_name)
        if args.run_test:
            with open(args_file, "r") as f:
                test_args = json.load(f)
            test_args["test_list"] = args.test_list
            test_args["save_results"] = args.save_results
            test_args["dataset_dir"] = args.dataset_dir
            test_args["experiment_name"] = experiment_folder
            test_args["gpu_ids"] = '0'
            test_args["load_normals"] = True
            test_args["load_planes"] = True
            report = test.test(args=SimpleNamespace(**test_args))
            # if args.save_results:
            #     visualizations.visualizePclDepth(test_results_folder)
        else:
            with open(osp.join(test_results_folder, "metrics.txt"), "r") as f:
                report = json.load(f)

        dict_1d = {
            "name": experiment_name
        }
        for cat_key, category in report.items():
            for key, val in category.items():
                dict_1d[key] = val
        experiments.append(dict_1d)

    df = pd.DataFrame.from_dict(experiments, orient='columns')
    df.to_csv(osp.join(args.experiments_folder, "results.csv"))


if __name__ == '__main__':
    main()
