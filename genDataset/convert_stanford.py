import utils
from PIL import Image
import os.path as osp
import numpy as np
import glob
import argparse
import os
import shutil
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Covert Stanford dataset to Omidepth dataset format')
    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument('result_dir', type=str,
                        help='All generated files will be saved here')
    args = parser.parse_args()

    if (not osp.exists(args.dataset)):
        print("Directory: {} does not exist. Skipping.".format(args.dataset))
        return
    os.makedirs(args.result_dir, exist_ok=True)
    depth_files = glob.glob(args.dataset + '/**/*_depth.png', recursive=True)
    print(depth_files)
    # return
    for i, depth_file in tqdm(enumerate(depth_files)):
        depth_folder, filename = osp.split(depth_file)
        base_folder, _ = osp.split(depth_folder)
        room_folder, area_name = osp.split(base_folder)
        image_folder = osp.join(base_folder, "rgb")

        rgb_filename = filename.replace("depth", "rgb")
        rgb_file = osp.join(image_folder, rgb_filename)

        result_folder = osp.join(args.result_dir, area_name)
        os.makedirs(result_folder, exist_ok=True)

        new_base_name = str(i) + "_" + area_name + "_" + filename[:-4].split("_")[1]
        new_rgb_file = osp.join(result_folder, new_base_name + "_color_0.png")
        new_depth_file = osp.join(result_folder, new_base_name + "_depth_0.tiff")
        final_size = (512, 256)

        color = Image.open(rgb_file).resize(final_size, Image.LANCZOS)
        color.save(new_rgb_file)

        color_np = np.sum(np.array(color)[:, :, 0:2], axis=2)
        # print((color_np == 0)[100:103, 100:103])
        # print(color_np, color_np.dtype, color_np.shape)
        color_mask = np.ones((256, 512))
        color_mask[color_np == 0] = 0

        depth = Image.open(depth_file).resize(final_size, Image.LINEAR)
        depth = np.array(depth).astype(float)
        depth[depth >= 2e16 - 2] = 0
        depth = depth / 512
        depth = depth * color_mask
        utils.write_tiff(new_depth_file, depth)


if __name__ == '__main__':
    main()
