import numpy as np
import cv2
import argparse
import glob
import os.path as op
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adds another column to existing test/train set. This columns has filenames of sparse depth files"
    )
    parser.add_argument("dataset", type=str, help="Path to dataset split file")
    parser.add_argument("output", type=str, help="Path to dataset split file")
    parser.add_argument("--distance", default=20, type=int,
                        help="Minimal distance between key-points in pixels")
    parser.add_argument("--max_points", default=100, type=int,
                        help="Number of extracted key-points")
    args = parser.parse_args()
    frame = pd.read_csv(args.dataset, header=None,
                        sep=" ", names=["color", "depth"])
    frame["points"] = frame["color"]
    frame["points"] = frame["points"].str.replace(
        pat="_color_", repl="_points_", n=1)
    frame["points"] = frame["points"].str.replace(
        pat=".png", repl='_D'+str(args.distance)+'_C'+str(args.max_points)+'.exr')
    # print(frame["points"][0])
    frame.to_csv(path_or_buf=args.output, sep=" ", header=None, index=None)
