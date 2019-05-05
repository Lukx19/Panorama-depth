from enum import Enum
from collections import defaultdict
import copy
import torch


class DataType(Enum):
    Image = 0
    Depth = 1
    Normals = 2
    PlanarSegmentation = 3
    Points3d = 4
    SparseDepth = 5
    Planes = 6
    Mask = 7
    PlaneParams = 8


class AnnotatedData:
    def __init__(self):
        self.filenames = []
        self._data = defaultdict(lambda: defaultdict(lambda: []))

    def add(self, tensor, data_type, scale=1):
        if type(scale) is not str:
            scale = str(scale)
        self._data[data_type][scale].append(tensor)
        return self

    def get(self, data_type, scale='1'):
        """Returns list of tensors of selected type and scale """
        if type(scale) is not str:
            scale = str(scale)
        return self._data[data_type][scale]

    def queryType(self, data_type):
        """Returns list of (scale,tensor) of selected type """
        res = []
        for scale, tensors in self._data[data_type].items():
            for tensor in tensors:
                res.append((scale, tensor))
        return res

    def remove(self, type, scale=1):
        raise NotImplementedError()
        pass

    def to(self, device):
        new_data = AnnotatedData()
        new_data.filenames = copy.deepcopy(self.filenames)
        for data_key, scale_dict in self._data.items():
            for scale_key, tensors in scale_dict.items():
                for tensor in tensors:
                    new_data.add(tensor.to(device), data_key, scale_key)
        return new_data

    def cpu(self):
        return self.to(torch.device('cpu'))
