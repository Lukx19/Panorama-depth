__author__ = "Lukas Jelinek"

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import torch
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image, ImageOps
import os.path as osp
from util import read_tiff
from annotated_data import DataType
from glob import glob


def pad_tensor(tensor, pad, dim):
    """
    args:
        tensor - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(tensor.shape)
    pad_size[dim] = pad - tensor.size(dim)
    return torch.cat([tensor, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the biggest number of plane masks in
    a batch of plane masks
    """

    def __init__(self, dim=1):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        print(batch[0], batch[1])
        # find the biggest number of planes
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = map(lambda x, y:
                    (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], batch))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


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
        if image == [] or image is None:
            raise Exception("Not able to convert None image to Tensor")
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


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return ImageOps.equalize(image)


#  converts to tensor and normalizes to [0,1] range
default_transformer = transforms.Compose([
    # Normalize(),
    ToTensor(scale=True),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
default_depth_transformer = transforms.Compose([ToTensor(scale=False)])
prediction_rgb_trasformer = transforms.Compose([
    # Normalize(),
    transforms.Resize(size=(256, 512)),
    ToTensor(scale=True)])

imagenet_transformer = transforms.Compose([
    ToTensor(scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class OmniDepthDataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for effiicient loading'''

    def __init__(self, root_path, path_to_img_list, use_sparse_pts=False,
                 use_normals=False, use_planes=False,
                 transformer_rgb=default_transformer, transformer_depth=default_depth_transformer,
                 transformer_points=default_depth_transformer):

        # Set up a reader to load the panos
        self.root_path = osp.abspath(root_path)

        # Create tuples of inputs/GT
        self.image_list = np.loadtxt(path_to_img_list, dtype=str)

        # Max depth for GT
        self.max_depth = 8.0
        self.use_sparse_pts = use_sparse_pts
        self.use_normals = use_normals
        self.use_planes = use_planes

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

        basename = osp.splitext(osp.basename(relative_paths[0]))[0]
        # print(basename)

        # read RGB convert to PIL and apply transformation
        original_rgb = Image.open(osp.join(self.root_path, relative_paths[0]))
        rgb = self.transformer_rgb(original_rgb)

        # read EXR convert to numpy and convert to PIL. Then apply transformation.
        depth = self.readDepthPano(osp.join(self.root_path, relative_paths[1]))
        depth_mask = ((depth <= self.max_depth) &
                      (depth > 0.)).astype(np.uint8)
        # Threshold depths
        depth *= depth_mask

        data = {}
        if self.use_sparse_pts:
            sparse_depth = self.readDepthPano(
                osp.join(self.root_path, relative_paths[2]))
            sparse_depth *= depth_mask
            sparse_depth /= self.max_depth

            sparse_depth = self.transformer_points(
                transforms.functional.to_pil_image(sparse_depth))
            data[DataType.SparseDepth] = sparse_depth

        depth = self.transformer_depth(
            transforms.functional.to_pil_image(depth))
        depth_mask = self.transformer_depth(
            transforms.functional.to_pil_image(depth_mask))

        if self.use_normals:
            normals_fname = relative_paths[1]
            normals_fname = normals_fname.replace("_depth_", "_normals_", 1)
            normals_fname = osp.join(self.root_path, normals_fname)
            normals = ToTensor()(self.readNormals(normals_fname))
            normals = normals * depth_mask
            data[DataType.Normals] = normals

        seg_fname = relative_paths[1]
        seg_fname = seg_fname.replace("_depth_", "_planes_", 1)
        seg_fname = osp.join(self.root_path, seg_fname)

        if self.use_planes:
            plane_seg = ToTensor()(self.readPlanarSegmentation(seg_fname))
            plane_instances = ToTensor(scale=False)(self.readPlanarInstances(seg_fname))
            data[DataType.PlanarSegmentation] = plane_seg
            data[DataType.Planes] = plane_instances

        data[DataType.Image] = rgb
        data[DataType.Depth] = depth
        data[DataType.Mask] = depth_mask

        return [data, basename]

    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.image_list)

    def readDepthPano(self, path):
        data = read_tiff(path).astype(np.float32)
        return np.reshape(data, (*data.shape, 1))

    def readNormals(self, path):
        if osp.exists(path):
            normals = read_tiff(path)
            return normals
        else:
            print("File missing", path)
            return []

    def readPlanarSegmentation(self, path):
        if osp.exists(path):
            planes = read_tiff(path)
            is_planar = 1 - planes[:, :, 0]

            return np.reshape(is_planar, (*is_planar.shape, 1))
        else:
            raise ValueError("File missing", path)

    def readPlanarInstances(self, path):
        if osp.exists(path):
            max_planes = 30
            # first plane are non planar pixels
            planes = read_tiff(path)[:, :, 1:]
            if len(planes.shape) == 2:
                print(path)
                h, w = planes.shape
                return np.zeros((h, w, max_planes))

            h, w, ch = planes.shape
            perm = np.argsort(np.sum(planes, axis=(0, 1)))
            perm = np.flip(perm)
            planes = planes[:, :, perm]
            # use only dominant 15 planes for now. We can increase it later
            if ch < max_planes:
                # print(planes.shape, path)
                if ch == 1:
                    planes = np.zeros((h, w, 1))
                planes = np.concatenate([planes, np.zeros((h, w, max_planes - ch))], axis=2)
            planes2 = planes[:, :, 0:max_planes]
            # print(np.sum(planes2, axis=(0, 1)), planes2.shape)

            return planes2
        else:
            raise ValueError("File missing", path)


class ImageDataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for effiicient loading'''

    def __init__(self, image_folder, transformer_rgb=default_transformer):

        # Create tuples of inputs/GT
        self.images = glob(image_folder + "/**/*.jpg", recursive=True)
        self.transformer_rgb = transformer_rgb

    def __getitem__(self, idx):
        '''Load the data'''
        basename = osp.splitext(osp.basename(self.images[idx]))[0]
        data = {}
        original_rgb = Image.open(self.images[idx])
        rgb = self.transformer_rgb(original_rgb)
        data[DataType.Image] = rgb
        _, h, w = rgb.size()
        data[DataType.SparseDepth] = torch.zeros((1, h, w))
        return [data, basename]

    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.images)
