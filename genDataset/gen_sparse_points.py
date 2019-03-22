import numpy as np
import cv2
import argparse
import glob
import os.path as op

def createNewFileName(base_dir,basename,extention, to_replace, replace_by,postfix=None):
    name_parts = basename.split("_")
    name_parts = list(map(lambda d: replace_by if d ==
                              to_replace else d, name_parts))
    name =  "_".join(name_parts)
    if postfix:
        name = name+ postfix

    return op.abspath(op.join(base_dir, name + extention))

def main():
    parser = argparse.ArgumentParser(description='Generate dataset with sparse depth images when provided with depth and rgb images')
    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument('--distance', default=20, type=int,
                        help='Minimal distance between key-points in pixels')
    parser.add_argument('--max_points', default=100, type=int,
                        help='Number of extracted key-points')
    args = parser.parse_args()

    if (not op.exists(args.dataset)):
        print("Directory: {} does not exist. Skipping.".format(args.dataset))
        return
    image_files = glob.glob(args.dataset+'/*/*color*', recursive=True)

    for image_file in image_files:
        dirPath, file = op.split(image_file)
        basename, ext = op.splitext(file)
        depth_file = createNewFileName(dirPath,basename,".exr","color","depth")

        # find corners
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(image=gray, maxCorners=args.max_points, minDistance=args.distance, qualityLevel=0.01)
        corners = np.int0(corners)

        #  filter only keypoint depths
        depth_mask = np.zeros(img.shape[0:2])
        for corner in corners:
            depth_mask[corner[0][1], corner[0][0]] = 1
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        sparse_depth = depth * depth_mask
        sparse_depth = sparse_depth.astype(np.float32)

        # save to new file with postfix
        postfix = "_D"+str(args.distance)+"_C"+str(args.max_points)
        sparse_file = createNewFileName(dirPath,basename,".exr","color","points",postfix=postfix)
        cv2.imwrite(sparse_file, sparse_depth)

        # print(sparse_depth[np.where(sparse_depth > 0,True,False)])
        print(sparse_file)



if __name__ == '__main__':
    main()
