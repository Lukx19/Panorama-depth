import os
import os.path as osp
import argparse
import json
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Generate list of files which have high difference between them.')
    parser.add_argument('file_a', type=str, help='Path to file with results from first experiment')
    parser.add_argument('file_b', type=str, help='Path to file with results from second experiment')
    parser.add_argument('test_split', type=str, help='Path to file with test split paths')
    parser.add_argument('result_list', type=str,
                        help='Resulting test file split ')
    parser.add_argument('--use_depth_acc', action="store_true", default=False,
                        help='Run network test')

    args = parser.parse_args()
    results = {}
    with open(args.file_a, "r") as f:
        list_a = json.load(f)
        for d1, filename, mean_angle in list_a:
            if args.use_depth_acc:
                val = d1
            else:
                val = mean_angle
            if ("color_0" in filename and "Left" not in filename) or "_0.0" in filename:
                results[filename] = val

    diff_and_name = []
    with open(args.file_b, "r") as f:
        list_b = json.load(f)
        for d1, filename, mean_angle in list_b:
            if args.use_depth_acc:
                val = d1
            else:
                val = mean_angle
            if filename in results and val < results[filename]:
                difference = abs(val - results[filename])
                diff_and_name.append((difference, filename))

    diff_and_name.sort(key=lambda v: v[0], reverse=True)
    lines = []
    num_elements = min(20, len(diff_and_name))
    for difference, filename in diff_and_name[:num_elements]:
        print(difference, filename)
        cmd = f'grep /{filename} "{args.test_split}"'
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p.wait()
        lines.append(str(output, 'utf-8'))

    with open(args.result_list, "w") as f:
        f.write("".join(lines))


if __name__ == '__main__':
    main()
