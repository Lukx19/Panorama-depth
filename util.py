import torch
import torch.nn as nn
import torchvision.transforms as tvt

import numpy as np

import os
import os.path as osp
import shutil
from skimage import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from graphviz import Digraph
import plotly.graph_objs as go
import plotly as py
from PIL import Image
import math


def mkdirs(path):
    '''Convenience function to make all intermediate folders in creating a directory'''
    os.makedirs(path, exist_ok=True)


def xavier_init(m):
    '''Provides Xavier initialization for the network weights and
    normally distributes batch norm params'''
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1) or (classname.find('ConvTranspose2d') != -1):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, filename):
    '''Saves a training checkpoints'''
    torch.save(state, filename)
    if is_best:
        basename = osp.basename(filename)  # File basename
        # Index where path ends and basename begins
        idx = filename.find(basename)
        # Copy the file to a different filename in the same directory
        shutil.copyfile(filename, osp.join(filename[:idx], 'model_best.pth'))


def load_partial_model(model, loaded_state_dict):
    '''Loaded a save model, even if the model is not a perfect match. This will run even if there \
        is are layers from the current network missing in the saved model.
    However, layers without a perfect match will be ignored.'''
    if isinstance(model, torch.nn.DataParallel):
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()
    pretrained = []
    ignored_layers = []
    for k, v in loaded_state_dict.items():
        #  all this hell is because DataParallel adds 'module.' to layer name conv.bias
        #  If trained on two gpus and testing is on one gpu then loading
        # checkpoint needs to remove 'module.'
        if k in model_dict:
            pretrained.append((k, v))
            continue

        if 'module.' in k:
            k = k[7:]
        if k in model_dict:
            pretrained.append((k, v))
        else:
            ignored_layers.append(k)

    pretrained_dict = dict(pretrained)
    print("ignored layers in partial model load:", ignored_layers)
    model_dict.update(pretrained_dict)

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)


def load_encoder_weights(model, checkpoint_path):
    if checkpoint_path is None:
        raise ValueError("Checkpoint path is None")

    checkpoint = torch.load(checkpoint_path)
    if "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
        encoder_layers = filter(lambda val: ("encoder" in val[0]
                                             or "input" in val[0]), weights.items())
        encoder_layers = dict(encoder_layers)
        print("Loaded encoder layers:", encoder_layers.keys())
        load_partial_model(model, encoder_layers)
    else:
        print("Not possible to load encoder weights. Missing state_dict key in checkpoint")


def load_optimizer(optimizer, loaded_optimizer_dict, device):
    '''Loads the saved state of the optimizer and puts it back on the GPU if necessary. \
        Similar to loading the partial model, this will load only the optimization \
        parameters that match the current parameterization.'''
    optimizer_dict = optimizer.state_dict()
    pretrained_dict = {k: v for k, v in loaded_optimizer_dict.items()
                       if k in optimizer_dict and k != 'param_groups'}
    optimizer_dict.update(pretrained_dict)
    optimizer.load_state_dict(optimizer_dict)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def set_caffe_param_mult(m, base_lr, base_weight_decay):
    '''Function that allows us to assign a LR multiplier of 2 and a decay \
        multiplier of 0 to the bias weights (which is common in Caffe)'''
    param_list = []
    for name, params in m.named_parameters():
        if name.find('bias') != -1:
            param_list.append({'params': params, 'lr': 2 *
                               base_lr, 'weight_decay': 0.0})
        else:
            param_list.append({'params': params, 'lr': base_lr,
                               'weight_decay': base_weight_decay})
    return param_list


def read_tiff(image_fpath):
    img = io.imread(image_fpath)
    return img


def write_tiff(image_fpath, data):
    if len(data.shape) > 2 and data.shape[2] > 3:
        raise Exception("write tiff can write only data with up to 3 channels")

    if image_fpath[-5:] != ".tiff":
        io.imsave(image_fpath + ".tiff", data, check_contrast=False, compress=6)
    else:
        io.imsave(image_fpath, data, check_contrast=False, compress=6)


def saveTensorDepth(filename, tensor, scale):
    img = tensor.cpu().numpy()
    img = np.squeeze(img)
    # img = np.transpose(img, (1, 2, 0))
    # print(img.shape)
    img *= scale
    write_tiff(filename, img)
    # img = Tt.functional.to_pil_image(img)
    # img = Image.fromarray(img, mode="I")
    # print(img)
    # img.save(filename)


def toDevice(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        res = [toDevice(sub_obj, device) for sub_obj in obj]
        if isinstance(obj, tuple):
            return tuple(res)
        else:
            return res
    elif isinstance(obj, dict):
        return type(obj)([(k, toDevice(tensor, device)) for k, tensor in obj.items()])
    else:
        raise Exception("toDevice is missing specialization for this type ", type(obj))


def uncolapseMask(tensor):
    '''
    Takes in Bx1xHxW tensor with integers representing individual classes.
    Returns list of tensors BxCxHxW where C is number of binary masks represented
    by indeces in original tensor
    '''
    pass


def plot_grad_flow(named_parameters, filename):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad and p.is_leaf
                and p.grad is not None and("bias" not in n)):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient',
                                                     'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig(filename)


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)


def register_hooks(var):
    '''
        x = Variable(torch.randn(10, 10), requires_grad=True)
        y = Variable(torch.randn(10, 10), requires_grad=True)

        z = x / (y * 0)
        z = z.sum() * 2
        get_dot = register_hooks(z)
        z.backward()
        dot = get_dot()
        dot.save('tmp.dot')
        Author: https://gist.github.com/apaszke/f93a377244be9bfcb96d3547b9bc424d
    '''
    fn_dict = {}

    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '(' + (', ').join(map(str, size)) + ')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot


def imageHeatmap(rgb, heatmap, title="", colorscheme="Reds", max_val=8):
    '''
    rgb Tensor 3xHxW
    heatmap Tensor HxW
    '''
    height = 2 * heatmap.size(0)
    width = 2 * heatmap.size(1)
    # image = tvt.functional.to_pil_image((rgb.detach() * 255).byte())
    real_max = torch.max(heatmap).item()
    heatmap = go.Heatmap(
        z=heatmap.detach().numpy(),
        colorscale=colorscheme,
        opacity=1,
        zmin=0,
        zmax=max_val,
    )

    steps = []
    for num in np.linspace(0, real_max, 50):
        step = dict(
            method='restyle',
            args=['zmax', num],
            label=str(np.around(num, decimals=2)),
        )
        steps.append(step)

    # TODO: no image under heatmap right now. No idea why
    # https://community.plot.ly/t/using-local-image-as-background-image/4381/3

    # TODO: current version of visdom has a bug which makes it difficult to show title
    # https://github.com/facebookresearch/visdom/pull/561
    # wait for release of bugfix then replace dict with go.Layout to call below
    layout = dict(
        height=height,
        width=width,
        title=title,
        sliders=[dict(
            pad={"t": 50},
            steps=steps
        )]
        # images=[go.layout.Image(
        #     visible=True,
        #     # x=0,
        #     # sizex=width,
        #     # y=0,
        #     # sizey=height,
        #     # xref="paper",
        #     # yref="paper",
        #     opacity=1.0,
        #     layer="below",
        #     source=image,
        #     sizing="stretch")]
    )
    return {'data': [heatmap], 'layout': layout}


def heatmapGrid(heatmaps, titles, title="", columns=2, colorscheme="Viridis"):
    """Creates grid of plotly heatmaps with titles.

    Parameters
    ----------
    heatmaps : list of np.arrays of shape HxW
    titles : list of str
    title : str, optional
        Title of the whole figure, by default ""
    columns : int, optional
        number of rows is deduced from heatmaps and columns, by default 2
    colorscheme : str, optional
        Plotly color-sheme name
    """

    rows = math.ceil(len(heatmaps) / columns)

    height = rows * heatmaps[0].shape[0]
    width = columns * heatmaps[0].shape[1]

    fig = py.tools.make_subplots(rows=rows, cols=columns, subplot_titles=titles, print_grid=False)
    grid_positions = list(np.ndindex((rows, columns)))[:len(heatmaps)]
    limit_max = np.max([np.max(tensor) for tensor in heatmaps])
    limit_min = np.min([np.min(tensor) for tensor in heatmaps])
    for heatmap, (row, col) in zip(heatmaps, grid_positions):
        fig.append_trace(go.Heatmap(
            z=heatmap,
            zmin=limit_min,
            zmax=limit_max,
            colorscale=colorscheme
        ), row + 1, col + 1)
    # TODO: current version of visdom has a bug which makes it difficult to show title
    # https://github.com/facebookresearch/visdom/pull/561
    # wait for release of bugfix then add title=title to call below
    fig['layout'].update(height=height, width=width)
    return fig


def stackVerticaly(tensor):
    """Stacks channels of the tensor vertically going from CH x H x W to CH*H x W

    Parameters
    ----------
    tensor : Tensor CH x H x W

    Returns
    -------
    Tensor CH*H x W
    """

    ch, h, w = tensor.size()
    tensor = tensor.reshape((1, -1, w))
    return tensor


def mergeInDict(dict_a, dict_b):
    for key, val in dict_b.items():
        if key not in dict_a:
            dict_a[key] = val
    return dict_a
