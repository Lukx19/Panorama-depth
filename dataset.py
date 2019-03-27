__author__ = "Lukas Jelinek"

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import torch
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np
from skimage import io
import OpenEXR
import Imath
import array
from PIL import Image

import math
import os.path as osp


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if scale=True.
    """

    def __init__(self, scale=True):
        self.scale = scale

    def __call__(self, image):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if self.scale:
            return transforms.functional.to_tensor(image)
        else:
            np_img = np.array(image, np.float32)
            if len(np_img.shape) == 2:
                np_img = np.reshape(np_img, (*np_img.shape, 1))
            # print(np_img.shape)
            image = torch.from_numpy(np_img)
            # print(image.shape)
            return image.permute(2, 0, 1)


#  converts to tensor and normalizes to [0,1] range
default_transformer = transforms.Compose([ToTensor(scale=True)])
default_depth_transformer = transforms.Compose([ToTensor(scale=False)])


class OmniDepthDataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for effiicient loading'''

    def __init__(self, root_path, path_to_img_list, use_sparse_pts=False, transformer_rgb=default_transformer, transformer_depth=default_depth_transformer, transformer_points=default_depth_transformer):

        # Set up a reader to load the panos
        self.root_path = osp.abspath(root_path)

        # Create tuples of inputs/GT
        self.image_list = np.loadtxt(path_to_img_list, dtype=str)

        # Max depth for GT
        self.max_depth = 8.0
        self.use_sparse_pts = use_sparse_pts
        self.transformer_rgb = transformer_rgb
        self.transformer_depth = transformer_depth
        self.transformer_points = transformer_points

    def __getitem__(self, idx):
        '''Load the data'''

        # Select the panos to load
        relative_paths = self.image_list[idx]

        # convert from absolute path to relative
        for i, path in enumerate(relative_paths):
            if path[0] == '/':
                relative_paths[i] = path[1:]

        relative_basename = osp.splitext((relative_paths[0]))[0]
        basename = osp.splitext(osp.basename(relative_paths[0]))[0]
        print(basename)

        # read RGB convert to PIL and apply transformation
        original_rgb = Image.open(osp.join(self.root_path, relative_paths[0]))
        rgb = self.transformer_rgb(original_rgb)

        # read EXR convert to numpy and convert to PIL. Then apply transformation.
        depth = self.readDepthPano(osp.join(self.root_path, relative_paths[1]))
        depth_mask = ((depth <= self.max_depth) &
                      (depth > 0.)).astype(np.uint8)
        # Threshold depths
        depth *= depth_mask

        sparse_depth = []
        if self.use_sparse_pts:
            sparse_depth = self.readDepthPano(
                osp.join(self.root_path, relative_paths[2]))
            sparse_depth *= depth_mask
            sparse_depth /= self.max_depth

            sparse_depth = self.transformer_points(
                transforms.functional.to_pil_image(sparse_depth))

        depth = self.transformer_depth(
            transforms.functional.to_pil_image(depth))
        depth_mask = self.transformer_depth(
            transforms.functional.to_pil_image(depth_mask))

        return {
            "image": rgb,
            # "paths": relative_paths,
            "gt": depth,
            "mask": depth_mask,
            "name": basename,
            "sparse_depth": sparse_depth
        }

    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.image_list)

    def readDepthPano(self, path):
        data = self.read_exr(path)[..., 0].astype(np.float32)
        return np.reshape(data, (*data.shape, 1))

    def read_exr(self, image_fpath):
        f = OpenEXR.InputFile(image_fpath)
        dw = f.header()['dataWindow']
        w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        im = np.empty((h, w, 3))

        # Read in the EXR
        n_channels = len(f.header()["channels"])
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        if n_channels == 3:
            channels = f.channels(["R", "G", "B"], FLOAT)
        else:
            channels = f.channels(["Y"], FLOAT)

        for i, channel in enumerate(channels):
            im[:, :, i] = np.reshape(array.array('f', channel), (h, w))
        return im
