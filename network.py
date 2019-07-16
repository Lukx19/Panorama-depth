# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
__author__ = "Marc Eder"
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models

from util import xavier_init
from annotated_data import AnnotatedData, DataType, Annotation
import numpy as np
from modules import SmoothConv, PlanarConv, Depth2Points, Points2Depths
# from mean_shift import Bin_Mean_Shift

# @inproceedings{cheng2018depth,
#   title={Learning Depth with Convolutional Spatial Propagation Network},
#   author={Cheng, Xinjing and Wang, Peng and Yang, Ruigang},
#   journal={arXiv preprint arXiv:1810.02695},
#   year={2018}
# }


class CSPN(nn.Module):

    def __init__(self):
        super(CSPN, self).__init__()

    def forward(self, guidance, blur_depth, sparse_depth):

        # normalize features
        gate1_w1_cmb = torch.abs(guidance.narrow(1, 0, 1))
        gate2_w1_cmb = torch.abs(guidance.narrow(1, 1, 1))
        gate3_w1_cmb = torch.abs(guidance.narrow(1, 2, 1))
        gate4_w1_cmb = torch.abs(guidance.narrow(1, 3, 1))
        gate5_w1_cmb = torch.abs(guidance.narrow(1, 4, 1))
        gate6_w1_cmb = torch.abs(guidance.narrow(1, 5, 1))
        gate7_w1_cmb = torch.abs(guidance.narrow(1, 6, 1))
        gate8_w1_cmb = torch.abs(guidance.narrow(1, 7, 1))

        if sparse_depth is None:
            b, _, w, h = blur_depth.size()
            sparse_depth = blur_depth.new_zeros((b, 1, w, h))

        sparse_mask = torch.sign(sparse_depth)

        result_depth = (1 - sparse_mask) * blur_depth.clone() + \
            sparse_mask * sparse_depth

        for i in range(16):
            # one propagation
            spn_kernel = 3
            elewise_max_gate1 = self.eight_way_propagation(
                gate1_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate2 = self.eight_way_propagation(
                gate2_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate3 = self.eight_way_propagation(
                gate3_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate4 = self.eight_way_propagation(
                gate4_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate5 = self.eight_way_propagation(
                gate5_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate6 = self.eight_way_propagation(
                gate6_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate7 = self.eight_way_propagation(
                gate7_w1_cmb, result_depth, spn_kernel)
            elewise_max_gate8 = self.eight_way_propagation(
                gate8_w1_cmb, result_depth, spn_kernel)

            result_depth = self.max_of_8_tensor(elewise_max_gate1, elewise_max_gate2,
                                                elewise_max_gate3, elewise_max_gate4,
                                                elewise_max_gate5, elewise_max_gate6,
                                                elewise_max_gate7, elewise_max_gate8)

            result_depth = (1 - sparse_mask) * \
                result_depth.clone() + sparse_mask * sparse_depth

        return result_depth

    def eight_way_propagation(self, weight_matrix, blur_matrix, kernel):
        kernel_half = int((kernel - 1) // 2)

        # [batch_size, channels, height, width] = weight_matrix.size()
        # self.avg_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel, stride=1,
        #  padding=kernel_half, bias=False)
        weight = blur_matrix.new_ones((1, 1, kernel, kernel))
        weight[0, 0, kernel_half, kernel_half] = 0

        # self.avg_conv.weight = nn.Parameter(weight)

        # for param in self.avg_conv.parameters():
        #     param.requires_grad = False

        # self.sum_conv = nn.Conv2d(in_channels=1, out_channels=1,
        #   kernel_size=kernel, stride=1, padding=kernel_half, bias=False)
        sum_weight = blur_matrix.new_ones((1, 1, kernel, kernel))
        # self.sum_conv.weight = nn.Parameter(sum_weight)

        # for param in self.sum_conv.parameters():
        #     param.requires_grad = False

        # weight_sum = self.sum_conv(weight_matrix)
        # avg_sum = self.avg_conv((weight_matrix*blur_matrix))

        # print(weight_matrix.size(),blur_matrix.size())

        weight_sum = F.conv2d(input=weight_matrix, weight=sum_weight, bias=None, stride=1,
                              padding=kernel_half)
        avg_sum = F.conv2d(input=(weight_matrix * blur_matrix), weight=weight, bias=None, stride=1,
                           padding=kernel_half)

        out = (torch.div(weight_matrix, weight_sum)) * blur_matrix + torch.div(avg_sum, weight_sum)
        return out

    def normalize_gate(self, guidance):
        gate1_x1_g1 = guidance.narrow(1, 0, 1)
        gate1_x1_g2 = guidance.narrow(1, 1, 1)
        gate1_x1_g1_abs = torch.abs(gate1_x1_g1)
        gate1_x1_g2_abs = torch.abs(gate1_x1_g2)
        elesum_gate1_x1 = torch.add(gate1_x1_g1_abs, gate1_x1_g2_abs)
        gate1_x1_g1_cmb = torch.div(gate1_x1_g1, elesum_gate1_x1)
        gate1_x1_g2_cmb = torch.div(gate1_x1_g2, elesum_gate1_x1)
        return gate1_x1_g1_cmb, gate1_x1_g2_cmb

    def max_of_4_tensor(self, element1, element2, element3, element4):
        max_element1_2 = torch.max(element1, element2)
        max_element3_4 = torch.max(element3, element4)
        return torch.max(max_element1_2, max_element3_4)

    def max_of_8_tensor(self, element1, element2,
                        element3, element4, element5,
                        element6, element7, element8):
        max_element1_2 = self.max_of_4_tensor(
            element1, element2, element3, element4)
        max_element3_4 = self.max_of_4_tensor(
            element5, element6, element7, element8)
        return torch.max(max_element1_2, max_element3_4)


def signToLabel(x):
    """Changes from tensor with B x 3 x H x W where each channels
    has data in range [-1,1] to class label representation
    B x H x W with class labels [0,7]

    Parameters
    ----------
    x : FloatTensor [-1,1]
    """
    #  range change from [-1,1] to either 0 if negative or 1 if >=0
    x = torch.sign(torch.sign(x) + 1)
    # batch_size = x.size(0)
    ones, twos, fours = torch.split(torch.ones_like(x), split_size_or_sections=1, dim=1)
    twos *= 2
    fours *= 4
    binary_conv = torch.cat((ones, twos, fours), dim=1)
    return torch.sum(x * binary_conv, dim=1).long()


def labelToSign(tensor):
    """
    Parameters
    ----------
    tensor : Tensor B x H x W
        Each value is in range [0,7]

    Returns
    -------
    Tensor B x 3 x H x W
        Values are -1 or 1 based on class
    """

    x = tensor.int()
    channels = []
    for i in range(3):
        res = x // 2
        channels.append(torch.fmod(x, 2))
        x = res
    signs = torch.cat(channels, dim=1)
    # going from  0 or 1 to -1 or 1
    signs = signs + (signs - 1)
    return signs.float()


def expSphere(x):
    """Calculates activation function similar to softmax with sqrt in denominator
        Compute:
                exp(x_i) / sqrt( sum( exp(x_j)**2 ) )
    Parameters
    ----------
    x : Tensor Bx3xHxW

    Returns
    -------
    Tensor Bx3xHxW
    """
    batchsize, _, _, _ = x.size()
    x = torch.exp(x - torch.max(x))
    # Trick for numeric stability (norm won't becomes 0)
    # x = x + torch.sign(x) * 1e-4
    x = x + 1e-6
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    norm = norm + (1 - torch.sign(torch.abs(norm)))

    norm = norm.detach()
    emb = x / norm

    return emb


class NormalsDeepEmbedding(nn.Module):
    def __init__(self, in_channels):

        self.decoder_norm0_0 = ConvTransposeELUBlock(
            in_channels, 256, 4, stride=2, padding=1)

        self.decoder_norm0_1 = ConvELUBlock(
            256, 256, 5, padding=2)

        self.decoder_norm1_0 = ConvTransposeELUBlock(
            256, 128, 4, stride=2, padding=1)

        self.decoder_norm1_1 = ConvELUBlock(
            128, 128, 5, padding=2)
        self.decoder_norm1_0 = ConvTransposeELUBlock(
            256, 128, 4, stride=2, padding=1)
        self.decoder_norm1_2 = ConvELUBlock(
            128, 64, 1)
        self.normal_prediction = ConvELUBlock(64, 3, 3, padding=1)

    def forward(self, x):
        decoder_norm0_0_out = self.decoder_norm0_0(x)
        decoder_norm0_1_out = self.decoder_norm0_1(decoder_norm0_0_out)
        decoder_norm1_0_out = self.decoder_norm1_0(decoder_norm0_1_out)
        decoder_norm1_1_out = self.decoder_norm1_1(decoder_norm1_0_out)
        decoder_norm1_2_out = self.decoder_norm1_2(decoder_norm1_1_out)
        pred_normals = self.normal_prediction(decoder_norm1_2_out)
        return F.tanh(pred_normals)


class NormalsEmbedding(nn.Module):
    def __init__(self, in_channels, activation=torch.tanh):
        super(NormalsEmbedding, self).__init__()
        self.cov1 = ConvELUBlock(in_channels + 3, 16, 3, padding=1)
        self.cov2 = ConvELUBlock(in_channels + 3, 16, 3, padding=2, dilation=2)
        self.emb = nn.Conv2d(32, 3, 1)
        self.activation = activation
        height = 256
        width = 512
        self.to3d = Depth2Points(height, width)
        self.dirs = nn.Parameter(-self.to3d(torch.ones((1, 1, height, width))), requires_grad=False)
        # self.coords = torch.from_numpy(np.mgrid[0:height, 0:width]).float()

        # self.coords = nn.Parameter(torch.reshape(self.coords, (1, 2, height, width)),
        #    requires_grad=False)

    def forward(self, x):
        b_size = x.size(0)
        with torch.no_grad():
            # coords = torch.cat([self.coords for i in range(b_size)], dim=0)
            # print(coords.size())
            dirs = torch.cat([self.dirs for i in range(b_size)], dim=0)
        x = torch.cat((dirs, x), dim=1)
        # print(x, x.size())
        out1 = self.cov1(x)
        out2 = self.cov2(x)
        out = self.emb(torch.cat((out1, out2), dim=1))
        if self.activation is not None:
            return self.activation(out)
        else:
            return out


class ExpSegNormals(nn.Module):
    def __init__(self, in_channels, normal_activation=expSphere):
        super(ExpSegNormals, self).__init__()
        self.normals_emb = NormalsEmbedding(in_channels, activation=normal_activation)
        self.cov1 = ConvELUBlock(64 + 3, 16, 3, padding=1)
        self.cov2 = ConvELUBlock(64 + 3, 16, 3, padding=2, dilation=2)
        self.class_pred = nn.Conv2d(32, 8, 1)
        height = 256
        width = 512
        self.to3d = Depth2Points(height, width)
        self.dirs = nn.Parameter(-self.to3d(torch.ones((1, 1, height, width))), requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.size()
        emb = self.normals_emb(x)
        with torch.no_grad():
            dirs = torch.cat([self.dirs for i in range(b)], dim=0)

        x = torch.cat((dirs, x), dim=1)
        out1 = self.cov1(x)
        out2 = self.cov2(x)
        classes = self.class_pred(torch.cat((out1, out2), dim=1))

        # normal inference
        labels = torch.argmax(classes.detach(), dim=1, keepdim=True)
        signs = labelToSign(labels)
        normals = emb * signs
        return {
            "normals": torch.reshape(normals, (b, 3, h, w)),
            "normals_embedding": emb,
            "normals_classes": classes
        }


rectnet_ops = {
    "dilation": [(2, 5), (5, 2), (2, 2)],
    "iterations": [10, 10, 5],
    "kernel_size": [(5, 7), (7, 5), (3, 3)],
    "smooth_treshold": 0.4,
    "reflection_pad": False,
    "cspn": False,
    "sigma_c_mult": [0.4, 0.4, 0.2],
    "sigma_n": [1 / 3, 1 / 3, 1 / 3],
    "guided_merge": True,
    "seg_small_merge": False,
    "fusion_merge": False,
    "use_group_norm": True,
}


class RectNet(nn.Module):

    def __init__(self, in_channels,
                 normal_est=False, segmentation_est=False,
                 calc_planes=False, normals_est_type='standart',
                 normal_smoothing=False, ops=rectnet_ops):
        super(RectNet, self).__init__()
        self.height = 256
        self.width = 512
        self.ops = ops
        reflection_pad = self.ops["reflection_pad"]
        self.use_gn = self.ops["use_group_norm"]
        # Network definition
        self.input0_0 = ConvELUBlock(in_channels, 8, (3, 9), padding=(
            1, 4), reflection_pad=reflection_pad, group_norm=self.use_gn)
        self.input0_1 = ConvELUBlock(in_channels, 8, (5, 11), padding=(
            2, 5), reflection_pad=reflection_pad, group_norm=self.use_gn)
        self.input0_2 = ConvELUBlock(in_channels, 8, (5, 7), padding=(
            2, 3), reflection_pad=reflection_pad, group_norm=self.use_gn)
        self.input0_3 = ConvELUBlock(
            in_channels, 8, 7, padding=3, reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.input1_0 = ConvELUBlock(32, 16, (3, 9), padding=(
            1, 4), reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.input1_1 = ConvELUBlock(32, 16, (3, 7), padding=(
            1, 3), reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.input1_2 = ConvELUBlock(32, 16, (3, 5), padding=(
            1, 2), reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.input1_3 = ConvELUBlock(
            32, 16, 5, padding=2, reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.encoder0_0 = ConvELUBlock(
            64, 128, 3, stride=2, padding=1, reflection_pad=reflection_pad, group_norm=self.use_gn)
        self.encoder0_1 = ConvELUBlock(
            128, 128, 3, padding=1, reflection_pad=reflection_pad, group_norm=self.use_gn)
        self.encoder0_2 = ConvELUBlock(
            128, 128, 3, padding=1, reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.encoder1_0 = ConvELUBlock(
            128, 256, 3, stride=2, padding=1, reflection_pad=reflection_pad, group_norm=self.use_gn)
        self.encoder1_1 = ConvELUBlock(
            256, 256, 3, padding=2, dilation=2, reflection_pad=reflection_pad,
            group_norm=self.use_gn)
        self.encoder1_2 = ConvELUBlock(
            256, 256, 3, padding=4, dilation=4, reflection_pad=reflection_pad,
            group_norm=self.use_gn)
        self.encoder1_3 = ConvELUBlock(
            512, 256, 1, reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.encoder2_0 = ConvELUBlock(
            256, 512, 3, padding=8, dilation=8, reflection_pad=reflection_pad,
            group_norm=self.use_gn)
        self.encoder2_1 = ConvELUBlock(
            512, 512, 3, padding=16, dilation=16, reflection_pad=reflection_pad,
            group_norm=self.use_gn)
        self.encoder2_2 = ConvELUBlock(
            1024, 512, 1, reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.decoder0_0 = ConvTransposeELUBlock(
            512, 256, 4, stride=2, padding=1, reflection_pad=reflection_pad)

        self.decoder0_1 = ConvELUBlock(
            256, 256, 5, padding=2, reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.prediction0 = nn.Conv2d(256, 1, 3, padding=1)

        self.decoder1_0 = ConvTransposeELUBlock(
            256, 128, 4, stride=2, padding=1, reflection_pad=reflection_pad)

        self.decoder1_1 = ConvELUBlock(
            128, 128, 5, padding=2, reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.decoder1_2 = ConvELUBlock(
            129, 64, 1, reflection_pad=reflection_pad, group_norm=self.use_gn)

        self.prediction1 = nn.Conv2d(64, 1, 3, padding=1)

        self.cspn = self.ops["cspn"]
        if self.cspn:
            self.guidance = ConvELUBlock(
                64, 8, 3, padding=1, reflection_pad=reflection_pad, group_norm=self.use_gn)
            self.cspn = CSPN()

        # self.depth_normals = DepthToNormals(height=self.height, width=self.width,
        #                                     kernel_size=3, padding=2, dilation=2)

        self.calc_normals = normal_est
        self.calc_planes = calc_planes
        self.calc_segmentation = segmentation_est
        self.normal_smoothing = normal_smoothing
        # self.plane_type = plane_type
        self.calc_norm_seg_2x = False
        self.calc_merge_guidance = self.ops["seg_small_merge"]
        self.fusion_merge = self.ops["fusion_merge"]

        if self.calc_planes:
            self.calc_normals = True
            self.calc_segmentation = True
            self.calc_norm_seg_2x = True
            self.normal_smoothing = False

        if self.normal_smoothing:
            self.calc_normals = True
            self.calc_norm_seg_2x = True
            self.calc_planes = False
            self.calc_segmentation = True

        if not self.calc_planes and not self.normal_smoothing:
            self.fusion_merge = False
            self.calc_merge_guidance = False

        self.normals_type = normals_est_type
        if self.calc_normals or self.calc_norm_seg_2x:
            self.decoder1_normal = ConvELUBlock(128, 64, 3, dilation=2, padding=2,
                                                reflection_pad=reflection_pad,
                                                group_norm=self.use_gn)
            if self.normals_type == "standart":
                self.normals = NormalsEmbedding(in_channels=64)
            elif self.normals_type == "sphere":
                self.normals = ExpSegNormals(in_channels=64)
            elif self.normals_type == "plane":
                self.normals = NormalsEmbedding(in_channels=64, activation=None)
            elif self.normals_type == "deep":
                self.normals = NormalsDeepEmbedding(in_channels=512)
            else:
                raise ValueError("Unknown type of normal prediction module")

        if self.calc_segmentation:
            self.decoder1_seg = ConvELUBlock(128, 64, 3, dilation=2, padding=2,
                                             reflection_pad=reflection_pad, group_norm=self.use_gn)
            self.decoder2_seg = nn.Conv2d(64, 1, 1)

        if self.calc_norm_seg_2x:
            #  need to get half spatial size of normals
            self.avg_pool_normal = nn.AvgPool2d(3, stride=2, padding=1)
            self.avg_pool_seg = nn.AvgPool2d(3, stride=2, padding=1)

        if self.normal_smoothing:
            with torch.no_grad():
                self.to3d = Depth2Points(self.height, self.width)
                self.to3d_2x = Depth2Points(self.height // 2, self.width // 2)
                self.d2pt_2x = Points2Depths(self.height // 2, self.width // 2)
                smooth_layers = []
                for kernel_size, dilation, iters, sigma_c_mult, sigma_n in zip(
                        self.ops["kernel_size"], self.ops["dilation"], self.ops["iterations"],
                        self.ops["sigma_c_mult"], self.ops["sigma_n"]):
                    kx, ky = kernel_size
                    dx, dy = dilation
                    px, py = (kx // 2 * dx), (ky // 2 * dy)
                    smooth_layers.append(SmoothConv(kernel_size=kernel_size, padding=(px, py),
                                                    dilation=(dx, dy), iterations=iters,
                                                    sigma_c_mult=sigma_c_mult,
                                                    sigma_n=sigma_n))

                self.smoothing_l = nn.ModuleList(smooth_layers)

        if self.calc_planes:
            self.planar_to_depth = PlanarToDepth(height=self.height // 2, width=self.width // 2)
            self.to3d_2x = Depth2Points(self.height // 2, self.width // 2)
            self.to3d = Depth2Points(self.height, self.width)
            # self.merge_conv_p = nn.Conv2d(1, 16, 3, padding=1)
            # self.merge_conv_d = nn.Conv2d(1, 16, 3, padding=1)
            # self.merge_conv = nn.Conv2d(32, 1, 3, padding=1)

            # self.prediction_planar = nn.Conv2d(66, 1, 3, padding=1)
            # self.mean_shift = Bin_Mean_Shift()
            # self.plane_emb = nn.Conv2d(4, 2, 1)
            # Initialize the network weights
        if self.calc_merge_guidance:
            self.guidance = nn.Conv2d(64, 1, 3, padding=1)
            self.guidance2 = nn.Conv2d(2, 1, 3, padding=1)

        if self.fusion_merge:
            self.fusion_d = FilterBlock(in_channels=1, reflection_pad=reflection_pad,
                                        group_norm=self.use_gn)
            self.fusion_g = FilterBlock(in_channels=1, reflection_pad=reflection_pad,
                                        group_norm=self.use_gn)
            self.fusion = FilterBlock(in_channels=2, reflection_pad=reflection_pad,
                                      group_norm=self.use_gn)

        self.apply(xavier_init)
        print(f""""normals: {self.calc_normals}, seg: {self.calc_segmentation},
              normals2x: {self.calc_norm_seg_2x}, smoothing: {self.normal_smoothing},
              planes: {self.calc_planes}""")

    def forward(self, inputs):
        outputs = {}
        outputs["pcl"] = []
        outputs["pcl2x"] = []

        x = inputs[DataType.Image]
        if DataType.SparseDepth in inputs:
            sparse_depth = inputs[DataType.SparseDepth][1]
            x = torch.cat((x, sparse_depth), 1)
        else:
            sparse_depth = None

        # First filter bank
        # print(x.size())
        input0_0_out = self.input0_0(x)
        input0_1_out = self.input0_1(x)
        input0_2_out = self.input0_2(x)
        input0_3_out = self.input0_3(x)
        # print("ss", input0_0_out.size())
        # print("ss", input0_1_out.size())
        # print("ss", input0_2_out.size())
        # print("ss", input0_3_out.size())
        input0_out_cat = torch.cat(
            (input0_0_out,
             input0_1_out,
             input0_2_out,
             input0_3_out), 1)

        # Second filter bank
        # print("ss", input0_out_cat.size())
        input1_0_out = self.input1_0(input0_out_cat)
        input1_1_out = self.input1_1(input0_out_cat)
        input1_2_out = self.input1_2(input0_out_cat)
        input1_3_out = self.input1_3(input0_out_cat)
        # print(input1_3_out.size())
        # First encoding block
        encoder0_0_out = self.encoder0_0(
            torch.cat((input1_0_out,
                       input1_1_out,
                       input1_2_out,
                       input1_3_out), 1))
        encoder0_1_out = self.encoder0_1(encoder0_0_out)
        encoder0_2_out = self.encoder0_2(encoder0_1_out)
        # print(encoder0_2_out.size())
        # Second encoding block
        encoder1_0_out = self.encoder1_0(encoder0_2_out)
        encoder1_1_out = self.encoder1_1(encoder1_0_out)
        encoder1_2_out = self.encoder1_2(encoder1_1_out)
        encoder1_3_out = self.encoder1_3(
            torch.cat((encoder1_1_out, encoder1_2_out), 1))
        # print(encoder1_3_out.size())
        # Third encoding block
        encoder2_0_out = self.encoder2_0(encoder1_3_out)
        encoder2_1_out = self.encoder2_1(encoder2_0_out)
        encoder2_2_out = self.encoder2_2(
            torch.cat((encoder2_0_out, encoder2_1_out), 1))
        # print(encoder2_2_out.size())
        # print("-------------------")
        # First decoding block
        decoder0_0_out = self.decoder0_0(encoder2_2_out)
        # print(decoder0_0_out.size())
        decoder0_1_out = self.decoder0_1(decoder0_0_out)
        # print(decoder0_1_out.size())
        # 2x downsampled prediction
        pred_2x = self.prediction0(decoder0_1_out)
        upsampled_pred_2x = F.interpolate(pred_2x.detach(), scale_factor=2, mode='bilinear')

        # Second decoding block
        decoder1_0_out = self.decoder1_0(decoder0_1_out)

        normals_2x = None
        seg_2x = None
        mask = inputs[DataType.Mask]
        if self.calc_normals:
            #  1. calculate normals with decoder from in 1x resolution
            normals_input = self.decoder1_normal(decoder1_0_out)
            if self.normals_type == "sphere":
                res = self.normals(normals_input)
                pred_normals = res["normals"]
                pred_normals_embed = res["normals_embedding"]
                outputs["normals_classes"] = res["normals_classes"]
            else:
                res = self.normals(normals_input)
                pred_normals = res
                pred_normals_embed = res

            outputs["normals_emb"] = pred_normals_embed
            outputs["normals"] = pred_normals

        if self.calc_segmentation:
            # 1.1 calculate 1x segmentation
            seg_out = self.decoder1_seg(decoder1_0_out)
            pred_seg = self.decoder2_seg(seg_out)
            outputs["segmentation"] = pred_seg
        if self.calc_norm_seg_2x:
            # 2. downsample to 1/2 resolution -> reduce noise
            normals_2x = self.avg_pool_normal(pred_normals).detach()
            seg_2x = self.avg_pool_seg(torch.sigmoid(pred_seg.clone())).detach()

        if self.normal_smoothing:
            # 3. smooth depth based on normals
            with torch.no_grad():
                mask2x = F.interpolate(mask, scale_factor=0.5, mode="nearest")
                hard_seg = torch.where(seg_2x > 0.5, torch.ones_like(seg_2x),
                                       torch.zeros_like(seg_2x))
                # hard_seg = torch.ones_like(seg_2x)
                hard_seg[:, :, 0:10, :] = 0
                hard_seg[:, :, -10:, :] = 0
                hard_seg = hard_seg * mask2x

                points_2x = self.to3d_2x(pred_2x.detach())
                outputs["pcl2x"].append(points_2x)
                smooth_pts = points_2x * hard_seg
                smooth_normals = hard_seg * normals_2x
                for module in self.smoothing_l:
                    smooth_pts = module(smooth_pts, smooth_normals, None, hard_seg)
                    # smooth_pts = smooth_pts * hard_seg
                    # smooth_normals = hard_seg * smooth_normals
                    outputs["pcl2x"].append(smooth_pts)

                # smooth_depth = self.d2pt(smooth_depth * hard_seg, smooth_pts, normals * hard_seg)
                smooth_depth_2x = self.d2pt_2x(pred_2x.detach(), smooth_pts, normals_2x)
                depth_diff = torch.abs(pred_2x - smooth_depth_2x)
                smooth_depth_2x = torch.where(depth_diff > self.ops["smooth_treshold"],
                                              pred_2x, smooth_depth_2x)
                # smooth_depth_2x = hard_seg * smooth_depth_2x + pred_2x.detach() * (1 - hard_seg)
                upsampled_planar_depth = F.interpolate(smooth_depth_2x.detach(),
                                                       scale_factor=2, mode="bilinear")
                # upsampled_pred_2x = upsampled_planar_depth
                outputs["pcl"].append(self.to3d(upsampled_planar_depth.clone().detach()))
            # print(upsampled_planar_depth.requires_grad)

        if self.calc_planes and DataType.Planes in inputs:
            planes = inputs[DataType.Planes]
            planes_2x = F.max_pool2d(planes, kernel_size=3, stride=2, padding=1)
            planar_depth_2x = self.planar_to_depth(pred_2x.detach(), planes_2x,
                                                   seg_2x.detach(), normals_2x.detach(),
                                                   robust=False)
            outputs["pcl2x"].append(self.to3d_2x(pred_2x.clone().detach()))
            depth_diff = torch.abs(pred_2x - planar_depth_2x)
            planar_depth_2x = torch.where(depth_diff > self.ops["smooth_treshold"],
                                          pred_2x, planar_depth_2x)
            outputs["pcl2x"].append(self.to3d_2x(planar_depth_2x.clone().detach()))

            # emb_depth = self.merge_conv_d(pred_2x)
            # emb_plane = self.merge_conv_p(planar_depth_2x)
            # refined_planar_2x = self.merge_conv(torch.cat((emb_depth, emb_plane), dim=1))

            # refined_planar_2x = planar_depth_2x
            upsampled_planar_depth = F.interpolate(planar_depth_2x, scale_factor=2,
                                                   mode="bilinear")
            outputs["pcl"].append(self.to3d(upsampled_planar_depth.clone().detach()))

        decoder1_1_out = self.decoder1_1(decoder1_0_out)
        decoder1_2_out = self.decoder1_2(
            torch.cat((upsampled_pred_2x.detach(), decoder1_1_out), 1))

        # Second prediction output (original scale)
        pred_1x = self.prediction1(decoder1_2_out)

        if self.cspn:
            guidance = self.guidance(decoder1_2_out)
            opt_depth = self.cspn(
                guidance=guidance, blur_depth=pred_1x, sparse_depth=sparse_depth)
        else:
            opt_depth = pred_1x

        outputs["depth1x"] = opt_depth
        outputs["depth2x"] = pred_2x

        if self.calc_merge_guidance:
            merge_seg = None
            if self.ops["guided_merge"]:
                outputs["guidance"] = []
                merge = torch.tanh(self.guidance(decoder1_2_out))
                outputs["guidance"].append(merge.clone().detach())
                merge = self.guidance2(torch.cat((merge, pred_seg), dim=1))
                merge_seg = torch.sigmoid(merge)
                outputs["guidance"].append(merge_seg.clone().detach())
            else:
                merge_seg = torch.sigmoid(pred_seg)
                merge_seg = torch.where(merge_seg > 0.5, torch.ones_like(merge_seg),
                                        torch.zeros_like(merge_seg))

            # nonplanar_seg = 1 - planar_seg
            # outputs["depth_planar"] = upsampled_planar_depth * merge_seg
            planar_depth = upsampled_planar_depth * merge_seg
            # outputs["depth_nonplanar"] = opt_depth * (1 - merge_seg)
            basic_depth = opt_depth * (1 - merge_seg)

            # outputs["pcl"].append(self.to3d(outputs["depth_planar"].clone().detach()))
            # outputs["pcl"].append(self.to3d(outputs["depth_nonplanar"].clone().detach()))

            refined_depth = planar_depth + basic_depth

            outputs["pcl"].append(self.to3d(refined_depth.clone().detach()))
            outputs["depth1x"] = refined_depth

        if self.fusion_merge:
            emb_d = self.fusion_d(opt_depth)
            emb_g = self.fusion_g(upsampled_planar_depth)
            feat_planar = torch.cat((emb_d, emb_g), dim=1)
            emb_f = self.fusion(feat_planar)
            refined_depth = emb_f + upsampled_planar_depth
            outputs["depth_ref"] = refined_depth
            outputs["guidance"] = [emb_d, emb_g, emb_f]

        # outputs["depth_normals"] = self.depth_normals(opt_depth)
        # if self.calc_normals and not self.normal_smoothing:
        #     if self.normal_smoothing:
        #         normals_input = self.decoder1_normal(decoder1_0_out)
        #     elif self.normals_type == 'deep':
        #         normals_input = encoder2_2_out
        #     else:
        #         normals_input = decoder1_2_out

        #     if self.normals_type == "sphere":
        #         res = self.normals(normals_input)
        #         pred_normals = res["normals"]
        #         pred_normals_embed = res["normals_embedding"]
        #         outputs["normals_classes"] = res["normals_classes"]
        #     else:
        #         res = self.normals(normals_input)
        #         pred_normals = res
        #         pred_normals_embed = res

        #     outputs["normals_emb"] = pred_normals_embed
        #     outputs["normals"] = pred_normals

        # if self.calc_segmentation and not self.normal_smoothing:
        #     seg1 = self.seg_cov1(decoder1_2_out)
        #     seg2 = self.seg_cov2(decoder1_2_out)
        #     seg_cat = torch.cat((seg1, seg2), 1)
        #     pred_seg = self.seg_cov3(seg_cat)
        #     outputs["segmentation"] = pred_seg

        # if (self.calc_normals and self.calc_planes
        #         and DataType.Planes in inputs and self.calc_segmentation
        #         and self.plane_type == "fusion"):
        #     planes = inputs[DataType.Planes]
        #     planar_depth = self.planar_to_depth(opt_depth.detach(), planes,
        #                                         pred_seg.detach(), pred_normals.detach())
        #     print("aaaaa")
        #     # outputs["depth_planar"] = planar_depth
        #     # feat_planar = torch.cat((planar_depth, opt_depth), dim=1)
        #     # residuum = torch.sigmoid(self.fusion(feat_planar))
        #     # refined_depth = planar_depth * residuum + (1 - residuum) * opt_depth
        #     # outputs["guidance"] = residuum
        #     # outputs["guidance2"] = torch.sigmoid(pred_seg) * residuum
        #     # outputs["depth_ref"] = refined_depth
        #     # outputs["depth1x"] = refined_depth

        #     emb_d = self.fusion_d(opt_depth)
        #     emb_g = self.fusion_g(planar_depth)
        #     feat_planar = torch.cat((emb_d, emb_g), dim=1)
        #     emb_f = self.fusion(feat_planar)
        #     refined_depth = emb_f + planar_depth
        #     outputs["depth_ref"] = refined_depth
        #     # outputs["guidance"] = [emb_d, emb_g, emb_f]

        return outputs

    def annotateOutput(self, outputs):
        data = AnnotatedData()
        if "normals" in outputs:
            data.add(outputs["normals"], DataType.Normals)

        if "normals2x" in outputs:
            data.add(outputs["normals2x"], DataType.Normals, scale=2)

        if "normals_classes" in outputs:
            data.add(outputs["normals_classes"], DataType.NormalsClass)

        if "normals_emb" in outputs:
            data.add(outputs["normals_emb"], DataType.NormalsEmbed)

        if "PlaneParams" in outputs:
            data.add(outputs["PlaneParams"], DataType.PlaneParams)

        if "segmentation" in outputs:
            data.add(outputs["segmentation"], DataType.PlanarSegmentation)
        if "depth_planar" in outputs:
            data.add(outputs["depth_planar"], DataType.Depth)
        if "depth_ref" in outputs:
            data.add(outputs["depth_ref"], DataType.Depth)
        if "depth1x_planar" in outputs:
            data.add(outputs["depth1x_planar"], DataType.Depth)

        if "depth1x_smooth" in outputs:
            data.add(outputs["depth1x_smooth"], DataType.Depth)

        data.add(outputs["depth1x"], DataType.Depth)
        data.add(outputs["depth2x"], DataType.Depth, scale=2)
        if "depth_normals" in outputs:
            data.add(outputs["depth_normals"], DataType.DepthNormals)

        if "depth_planar" in outputs:
            data.add(outputs["depth_planar"], DataType.Depth, scale=1,
                     annotation=Annotation.PlanarBranch)

        if "depth_nonplanar" in outputs:
            data.add(outputs["depth_nonplanar"], DataType.Depth, scale=1,
                     annotation=Annotation.NonPlanarBranch)

        if "guidance" in outputs:
            for guidance in outputs["guidance"]:
                data.add(guidance, DataType.Guidance)

        if "pcl" in outputs:
            for pcl in outputs["pcl"]:
                data.add(pcl.detach(), DataType.Points3d)

        if "pcl2x" in outputs:
            for pcl in outputs["pcl2x"]:
                data.add(pcl.detach(), DataType.Points3d, scale=2)

        return data


# -----------------------------------------------------------------------------


class UResNet(nn.Module):

    def __init__(self, in_channels):
        super(UResNet, self).__init__()
        self.input0 = ConvELUBlock(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=7,
            stride=1,
            padding=3)
        self.input1 = ConvELUBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2)

        self.encoder0 = SkipBlock(64, 128)
        self.encoder1 = SkipBlock(128, 256)
        self.encoder2 = SkipBlock(256, 512)
        self.encoder3 = SkipBlock(512, 1024)

        self.decoder0_0 = ConvTransposeELUBlock(
            in_channels=1024,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1)
        self.decoder0_1 = ConvELUBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=5,
            stride=1,
            padding=2)
        self.decoder1_0 = ConvTransposeELUBlock(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1)
        self.decoder1_1 = ConvELUBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2)
        self.decoder2_0 = ConvTransposeELUBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1)
        self.decoder2_1 = ConvELUBlock(
            in_channels=128 + 1,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=2)
        self.decoder3_0 = ConvTransposeELUBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1)
        self.decoder3_1 = ConvELUBlock(
            in_channels=64 + 1,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2)

        self.prediction0 = nn.Conv2d(256, 1, 3, padding=1)
        self.prediction1 = nn.Conv2d(128, 1, 3, padding=1)
        self.prediction2 = nn.Conv2d(64, 1, 3, padding=1)

        self.apply(xavier_init)

    def forward(self, inputs):
        x = inputs[DataType.Image]
        if DataType.SparseDepth in inputs:
            sparse_depth = inputs[DataType.SparseDepth][1]
            x = torch.cat((x, sparse_depth), 1)
        else:
            sparse_depth = None
        # Encode down to 4x
        x = self.input0(x)
        x = self.input1(x)

        x = self.encoder0(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        x = self.decoder0_0(x)
        x = self.decoder0_1(x)
        x = self.decoder1_0(x)
        x = self.decoder1_1(x)

        # Predict at 4x downsampled
        pred_4x = self.prediction0(x)

        # Upsample through convolution to 2x
        x = self.decoder2_0(x)
        upsampled_pred_4x = F.interpolate(pred_4x.detach(), scale_factor=2)

        # Predict at 2x downsampled
        x = self.decoder2_1(torch.cat((x, upsampled_pred_4x), 1))
        pred_2x = self.prediction1(x)

        # Upsample through convolution to 1x
        x = self.decoder3_0(x)
        upsampled_pred_2x = F.interpolate(pred_2x.detach(), scale_factor=2)

        # Predict at 1x
        x = self.decoder3_1(torch.cat((x, upsampled_pred_2x), 1))
        pred_1x = self.prediction2(x)

        return [pred_1x, pred_2x, pred_4x]

    def annotateOutput(self, outputs):
        data = AnnotatedData()
        data.add(outputs[0], DataType.Depth)
        data.add(outputs[1], DataType.Depth, scale=2)
        data.add(outputs[2], DataType.Depth, scale=4)
        return data


class DoubleBranchNet(nn.Module):

    def __init__(self, in_channels):
        super(DoubleBranchNet, self).__init__()
        self.use_shuffle = False
        if in_channels == 4:
            addition = 1
            self.use_sparse_pts = True
        else:
            addition = 0
            self.use_sparse_pts = False
        # 256 x 512
        self.input0_0 = ConvELUBlock(in_channels, 8, (3, 9), padding=(1, 4))
        self.input0_1 = ConvELUBlock(in_channels, 8, (5, 11), padding=(2, 5))
        self.input0_2 = ConvELUBlock(in_channels, 8, (5, 7), padding=(2, 3))
        self.input0_3 = ConvELUBlock(in_channels, 8, 7, padding=3)

        # 256 x 512
        self.input1_0 = ConvELUBlock(32, 16, (3, 9), padding=(1, 4))
        self.input1_1 = ConvELUBlock(32, 16, (3, 7), padding=(1, 3))
        self.input1_2 = ConvELUBlock(32, 16, (3, 5), padding=(1, 2))
        self.input1_3 = ConvELUBlock(32, 16, 5, padding=2)

        # 256 x 512

        self.encoder0_0 = ConvELUBlock(64, 128, 3, stride=2, padding=1)
        self.encoder0_1 = ConvELUBlock(128, 128, 3, padding=1)
        self.encoder0_2 = ConvELUBlock(128, 128, 3, padding=1)

        # 128 x 512

        self.encoder1_0 = ConvELUBlock(128, 256, 3, stride=2, padding=1)
        self.encoder1_1 = ConvELUBlock(256, 256, 3, padding=2, dilation=2)
        self.encoder1_2 = ConvELUBlock(256, 256, 3, padding=4, dilation=4)
        self.encoder1_3 = ConvELUBlock(512, 256, 1)

        # 64 x 128

        self.encoder2_0 = ConvELUBlock(256, 512, 3, padding=8, dilation=8)
        self.encoder2_1 = ConvELUBlock(512, 512, 3, padding=16, dilation=16)
        self.encoder2_2 = ConvELUBlock(1024, 512, 1)

        # 64 x 128

        self.decode_lvl3 = UpsampleShuffleBlock(
            768 + addition, 512, upscale=2, use_shuffle=self.use_shuffle)
        self.decode_conv_lvl3 = nn.Conv2d(768 + addition, 1, 1, padding=0)
        self.decode_bil_lvl3 = nn.UpsamplingBilinear2d(scale_factor=2)

        # 128 x 512 x 128

        self.decode_lvl2 = UpsampleShuffleBlock(
            256 + addition, 128, upscale=2, use_shuffle=self.use_shuffle)
        self.decode_conv_lvl2 = nn.Conv2d(256 + addition, 1, 1, padding=0)
        self.decode_bil_lvl2 = nn.UpsamplingBilinear2d(scale_factor=2)

        # 256 x 512 x 32

        self.decode_conv_lvl1 = nn.Conv2d(96 + addition, 1, 1, padding=0)
        # self.decode_lvl1 = UpsampleShuffleBlock(
        #     129, 128, upscale=2, use_shuffle=True)
        # self.decode_conv_lvl1 = nn.Conv2d(129, 1, 1, padding=0)
        # self.decode_bil_lvl1 = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.decode_lvl1 = UpsampleShuffleBlock(
        #     64, 128, upscale=2, use_shuffle=True)
        # self.decode_conv_lvl1 = nn.Conv2d(129, 1, 1, padding=0)
        # self.decode_bil_lvl1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.apply(xavier_init)

    def forward(self, inputs):
        x = inputs[DataType.Image]
        if DataType.SparseDepth in inputs:
            pts1x = inputs[DataType.SparseDepth][1]
            pts2x = inputs[DataType.SparseDepth][2]
            pts4x = inputs[DataType.SparseDepth][4]
            x = torch.cat((x, pts1x), 1)

        # upsampled_pred_4x = F.interpolate(
        #     pred_4x.detach(), scale_factor=2, mode='bilinear')
        # First filter bank
        input0_0_out = self.input0_0(x)
        input0_1_out = self.input0_1(x)
        input0_2_out = self.input0_2(x)
        input0_3_out = self.input0_3(x)
        layer0 = torch.cat(
            (input0_0_out,
             input0_1_out,
             input0_2_out,
             input0_3_out), 1)

        # Second filter bank
        input1_0_out = self.input1_0(layer0)
        input1_3_out = self.input1_3(layer0)
        input1_1_out = self.input1_1(layer0)
        input1_2_out = self.input1_2(layer0)

        layer1 = torch.cat((input1_0_out,
                            input1_1_out,
                            input1_2_out,
                            input1_3_out), 1)
        # First encoding block
        encoder0_0_out = self.encoder0_0(layer1)
        encoder0_1_out = self.encoder0_1(encoder0_0_out)
        layer2 = self.encoder0_2(encoder0_1_out)

        # Second encoding block
        encoder1_0_out = self.encoder1_0(layer2)
        encoder1_1_out = self.encoder1_1(encoder1_0_out)
        encoder1_2_out = self.encoder1_2(encoder1_1_out)
        encoder1_3_out = self.encoder1_3(
            torch.cat((encoder1_1_out, encoder1_2_out), 1))

        encoder2_0_out = self.encoder2_0(encoder1_3_out)
        encoder2_1_out = self.encoder2_1(encoder2_0_out)
        encoder2_2_out = self.encoder2_2(
            torch.cat((encoder2_0_out, encoder2_1_out), 1))

        if self.use_sparse_pts:
            cat_lvl3 = torch.cat((encoder1_3_out, encoder2_2_out, pts4x), 1)
        else:
            cat_lvl3 = torch.cat((encoder1_3_out, encoder2_2_out), 1)
        shuffle3_out = self.decode_lvl3(cat_lvl3)
        # print(shuffle3_out.size())
        res_scale4x = self.decode_conv_lvl3(cat_lvl3)
        # print(res_scale4x.size())
        bilin3_out = self.decode_bil_lvl3(res_scale4x)
        if self.use_sparse_pts:
            cat_lvl2 = torch.cat((layer2, shuffle3_out, pts2x), 1)
        else:
            cat_lvl2 = torch.cat((layer2, shuffle3_out), 1)
        # print(layer2.size(), shuffle3_out.size())
        shuffle2_out = self.decode_lvl2(cat_lvl2)
        # print("lllll", shuffle2_out.size())
        conv2_out = self.decode_conv_lvl2(cat_lvl2)
        res_scale2x = bilin3_out + conv2_out
        bilin2_out = self.decode_bil_lvl2(res_scale2x)
        if self.use_sparse_pts:
            cat_lvl1 = torch.cat((layer1, shuffle2_out, pts1x), 1)
        else:
            cat_lvl1 = torch.cat((layer1, shuffle2_out), 1)

        conv1_out = self.decode_conv_lvl1(cat_lvl1)
        res_scale1x = conv1_out + bilin2_out
        return [res_scale1x, res_scale2x, res_scale4x]

    def annotateOutput(self, outputs):
        data = AnnotatedData()
        data.add(outputs[0], DataType.Depth)
        data.add(outputs[1], DataType.Depth, scale=2)
        data.add(outputs[2], DataType.Depth, scale=4)
        return data

# -----------------------------------------------------------------------------


class FilterBlock(nn.Module):
    def __init__(self, in_channels, reflection_pad=False, group_norm=True):
        super(FilterBlock, self).__init__()
        self.block = nn.Sequential(
            ConvELUBlock(in_channels, 64, 5, padding=2,
                         reflection_pad=reflection_pad, group_norm=group_norm),
            ConvELUBlock(64, 32, 1, padding=0,
                         reflection_pad=reflection_pad, group_norm=group_norm),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class CircularPad1d(nn.Module):
    def __init__(self, padding):
        super(CircularPad1d, self).__init__()
        self.left, self.right = 0, 0
        if type(padding) is int and padding > 0:
            self.left, self.right = padding, padding
        elif type(padding) is tuple:
            self.left, self.right = padding
        else:
            raise AttributeError("padding is neither tuple nor int")

    def forward(self, x):
        # x = torch.cat([x[:, :, -self.top:], x, x[:, :, 0:self.bottom]], dim=2)
        return torch.cat([x[:, :, :, -self.left:], x, x[:, :, :, 0:self.right]], dim=3)


def createPadding(padding):
    #  return new padding size and padding layer if padding is valid.
    # If padding is 0 then returns None padding layer
    reflection_pad = True
    if type(padding) is tuple and len(padding) > 1:
        h, w = padding
        pad_width = w
        pad_height = h
    else:
        if padding == 0:
            reflection_pad = False
        pad_width = padding
        pad_height = padding
    # print(padding, pad_height, pad_width)
    padding_l = None

    if reflection_pad:
        padding_l = nn.Sequential(
            CircularPad1d(
                padding=(pad_width, pad_width)),
            nn.ZeroPad2d(padding=(0, 0, pad_height, pad_height))
        )
        padding = 0

    return padding_l, padding


class ConvELUBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 reflection_pad=False,
                 group_norm=True):
        super(ConvELUBlock, self).__init__()
        self.reflection_pad = reflection_pad
        self.padding = padding
        if self.reflection_pad:
            self.padding_l, self.padding = createPadding(padding)
            if self.padding_l is None:
                self.reflection_pad = False
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation)
        self.group_norm = group_norm
        if self.group_norm:
            self.norm = nn.GroupNorm(num_groups=out_channels // 2, num_channels=out_channels)

    def forward(self, x):
        if self.reflection_pad:
            x = self.padding_l(x)
        if self.group_norm:
            return F.elu(self.norm(self.conv(x)), inplace=True)
        else:
            return F.elu(self.conv(x), inplace=True)


class UpsampleShuffleBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 upscale=2,
                 use_shuffle=True
                 ):
        super(UpsampleShuffleBlock, self).__init__()

        self.use_shuffle = use_shuffle
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1)
        self.bnorm = nn.BatchNorm2d(num_features=out_channels)
        if use_shuffle:
            self.upscale = nn.PixelShuffle(upscale_factor=upscale)
        else:
            self.upscale = nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels // 4,
                kernel_size=4,
                stride=2,
                padding=1,
                dilation=1)

    def forward(self, x):
        x = F.relu(self.bnorm(self.conv(x)), inplace=True)
        if self.use_shuffle:
            x = self.upscale(x)
        else:
            x = F.elu(self.upscale(x), inplace=True)
        return x


# -----------------------------------------------------------------------------
class ConvTransposeELUBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 reflection_pad=False):
        super(ConvTransposeELUBlock, self).__init__()

        self.reflection_pad = reflection_pad
        self.padding = padding
        self.reflection_pad = False
        if self.reflection_pad:
            self.padding_l, self.padding = createPadding(padding)
            if self.padding_l is None:
                self.reflection_pad = False
        # print('*********', padding, " aaa", self.padding, "aa ", self.reflection_pad)

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation)

    def forward(self, x):
        if self.reflection_pad:
            return F.elu(self.conv(self.padding_l(x)), inplace=True)
        return F.elu(self.conv(x), inplace=True)


# -----------------------------------------------------------------------------
class SkipBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SkipBlock, self).__init__()

        self.conv1 = ConvELUBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1)
        self.conv2 = ConvELUBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv3 = ConvELUBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):

        # First convolutional block
        out1 = self.conv1(x)

        # Second and third convolutional blocks
        out3 = self.conv3(self.conv2(out1))

        # Return the sum of the outputs of the first block and the third block
        return out1 + out3


def planeAwarePooling(params, planes, keepdim=False):
    '''
        params: B x 1 x 3 x N
        planes: B x P x 1 x N
        Where B is batch number, P is number of planes. Some planes might be empty
        and N is image height x width
        Return B x P x 3 x 1 or B x P x 3 x N if keepdim=True
    '''
    planes3_channel = None
    plane_weights = None
    with torch.no_grad():
        # b x p
        plane_weights = torch.sum(planes, dim=3) + 1
        planes3_channel = torch.cat((planes, planes, planes), dim=2)
    assert(not torch.isnan(params).any())
    # print(params.size(), planes.size(), planes3_channel.size())
    avg_param = torch.sum(params * planes3_channel, dim=3)
    # print(avg_param.size(), plane_weights.size())
    # b x p x 3
    avg_param /= plane_weights
    assert(not torch.isnan(avg_param).any())
    avg_param = torch.reshape(avg_param, (*avg_param.size(), 1))
    if keepdim:
        return avg_param * planes3_channel
    else:
        return avg_param


class PlanarToDepth(nn.Module):
    def __init__(self, height, width):
        super(PlanarToDepth, self).__init__()
        self.to3d = Depth2Points(height, width)

    def normalizeRange(self, depth, abs_range=8.0):
        with torch.no_grad():
            valid_pixels = torch.ones_like(depth)
            valid_pixels[depth > abs_range] = 0.0
            valid_pixels[depth < -abs_range] = 0.0
        ranged_depths = depth * valid_pixels
        return ranged_depths

    def forward(self, depth, planes, segmentation, normals, robust=True):
        points = self.to3d(depth)
        rays = self.to3d(torch.ones_like(depth))
        eps = 0.000001
        treshold = 0.8
        # planes = planes[:, 0:3, :, :]
        b, p, h, w = planes.size()
        planes_r = torch.reshape(planes, (b, p, 1, -1))
        points = torch.reshape(points, (b, 1, 3, -1))
        # depths = torch.reshape(depth, (b, 1, 1, -1))
        rays = torch.reshape(rays, (b, 1, 3, -1))
        normals = torch.reshape(normals, (b, 1, 3, -1))
        # normals += torch.isnan(normals.detach()).float()
        # BxPx3xN
        avg_normals = planeAwarePooling(normals, planes_r, keepdim=True)
        centroids = planeAwarePooling(points, planes_r, keepdim=True)
        if robust:
            # BxPx1xN
            cos_angle = torch.sum(avg_normals * normals, dim=2, keepdim=True)
            normal_mask = torch.where(cos_angle > treshold,
                                      torch.ones_like(planes_r), torch.zeros_like(planes_r))
            # Bx1x1xN
            normal_mask = torch.sum(normal_mask, dim=1, keepdim=True)
            avg_normals = planeAwarePooling(normal_mask * normals,
                                            normal_mask * planes_r, keepdim=True)

        # calculate distance to plane from camera center
        planes_bin = torch.sign(planes_r)
        norm = torch.sum(rays * (avg_normals.detach()), dim=2, keepdim=True)
        # print(norm.size(), planes_r.size(), rays.size(), torch.min(norm))
        norm = norm + (1 - torch.sign(planes_r.detach())) * eps

        centroid_normals = planes_bin * centroids * avg_normals
        planar_depths = torch.sum(centroid_normals, dim=2, keepdim=True) * planes_bin
        planar_depths = self.normalizeRange(planar_depths)

        planar_depths = planar_depths / norm
        nan_count = (1 - torch.isfinite(planar_depths)).sum().item()
        if nan_count > 0:
            print("found # inifinst", nan_count)
            planar_depths = torch.where(torch.isfinite(planar_depths),
                                        planar_depths, torch.zeros_like(planar_depths))
        # assert(not torch.isnan(planar_depths).any())
        # assert torch.isfinite(planar_depths).all()

        # planar_depths = torch.abs(planar_depths) * planes_bin
        planar_depths = planar_depths * planes_bin
        # assert torch.isfinite(planar_depths).all()
        planar_depths = torch.reshape(planar_depths, (b, p, -1))
        # assert(not torch.isnan(planar_depths).any())
        # print("negative:", torch.sum(planar_depths > eps), torch.sum(planar_depths < -eps))
        # print(planar_depths.size(), torch.min(planar_depths), torch.max(planar_depths))
        # assert(torch.min(planar_depths) >= 0)
        # B x P x N
        # plane_pixel_probs = torch.sum(planes_r.detach(), dim=(1, 2))
        # plane_pixel_probs = plane_pixel_probs + torch.sign(plane_pixel_probs) * 1e-4
        planar_depths = torch.sum(planar_depths, dim=1)
        # planar_depths = planar_depths / plane_pixel_probs
        planar_depths = torch.reshape(planar_depths, (b, 1, h, w))
        valid_pixels = torch.ones_like(planar_depths)
        valid_pixels[planar_depths > 8.0] = 0.0
        valid_pixels[planar_depths < 0.0] = 0.0
        # print(torch.isnan(planar_depths).sum())
        # valid_pixels[torch.isnan(planar_depths)] = 0
        # print(torcPh.isnan(planar_depths).sum())
        # planar_depths = torch.clamp(planar_depths, 0.0, 8.0)
        # seg = torch.sigmoid(segmentation)
        # seg = torch.sign(torch.sum(planes.detach(), dim=1, keepdim=True))
        # seg = torch.sign(planar_depths.detach())
        seg_planar = torch.sigmoid(segmentation)
        valid_pixels[seg_planar < 0.5] = 0

        # assert(not torch.isnan(planar_depths).any())
        planar_depths = planar_depths * valid_pixels
        # planar_depths += torch.isnan(planar_depths.detach()).float()
        # seg = torch.sign(planar_depths.clone().detach())
        # depth_res = seg * planar_depths + (1 - seg) * depth
        # assert(not torch.isnan(valid_pixels).any())
        # assert(not torch.isnan(seg).any())
        # assert(not torch.isnan(depth).any())
        # assert(not torch.isnan(depth_res).any())
        # print(depth_res.size(), seg.size(), planar_depths.size(), depth.size())
        return planar_depths


def toPlaneParams(normals, depth):
    '''
        normals: B x 3 x H x W
        depth: B x 1 x H x W
        Return plane parameter for each pixel B x 3 x H x W
    '''
    zero_mask = (depth.detach() < 0.00001).float()
    depth_padded = depth + torch.ones_like(depth) * zero_mask
    params = normals / depth_padded
    params = params * (1 - zero_mask)
    return params


class DepthToNormals(nn.Module):
    def __init__(self, height, width, kernel_size, dilation=1, padding=0, estimate_type="lin_sq"):
        super(DepthToNormals, self).__init__()
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation,
                                      padding=padding, stride=1)
        # self.fold = torch.nn.Fold(output_size=(height, width), kerne_size=kernel_size,
        #                           dilation=dilation, padding=padding, stride=1)
        self.height = height
        self.width = width
        self.to3d = Depth2Points(height, width)
        self.estimate_type = estimate_type
        if type(kernel_size) is int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        h, w = self.kernel_size
        self.ones = nn.Parameter(torch.ones((h * w, 1), requires_grad=False))
        self.eye = nn.Parameter(torch.eye(3), requires_grad=False)

    def batchedInv(self, tensor):
        orig_shape = tensor.size()
        tensor = tensor.reshape(-1, orig_shape[-2], orig_shape[-1])
        if tensor.shape[0] >= 256 * 256 - 1:
            temp = []
            for t in torch.split(tensor, 256 * 256 - 1):
                temp.append(torch.inverse(t))
            inverted = torch.cat(temp)
        else:
            inverted = torch.inverse(tensor)
        inverted = torch.reshape(inverted, orig_shape)
        return inverted

    def solve(self, points):
        """Uses An=b solution to estimate best fitting plane to 3d points for normal estimation

        Parameters
        ----------
        points : Tensor B x N x L of 3D points
            B batch size, N = 3xKhxKw whre Kh and Kw are kernel height and kernel width
            L are all kernel positions
        Returns
        -------
        Tensor Bx3xHxW
            Calculated normal for each pixel
        """

        b, n, k = points.size()
        Kh, Kw = self.kernel_size
        points = torch.transpose(points, dim0=1, dim1=2)
        # BxLxN
        A_t = torch.reshape(points, (b, k, 3, Kh * Kw))
        A = torch.transpose(A_t, dim0=2, dim1=3)
        A_t_A = torch.matmul(A_t, A)
        A_t_A = A_t_A + 1e-05 * self.eye
        A_t_o = torch.matmul(A_t, self.ones)
        A_t_A_inv = self.batchedInv(A_t_A)

        normals = torch.matmul(A_t_A_inv, A_t_o)
        # print(normals.size())
        norm = torch.norm(normals, p=2, dim=2, keepdim=True)
        norm = norm + (1 - torch.sign(torch.abs(norm))) * 1e-4
        # BxLx3x1
        # print(normals.size())
        normals = -normals / norm.detach()
        assert not torch.isnan(normals).any(), "Normal solving produces NaNs"
        normals = torch.transpose(normals, dim0=1, dim1=2)
        normals = torch.reshape(normals, (b, 3, self.height, self.width))
        return normals
        # solve An = b

    def forward(self, depth):
        points = self.to3d(depth)
        unfolded = self.unfold(points)
        if self.estimate_type == "lin_sq":
            normals = self.solve(unfolded)
        else:
            raise ValueError("Unknown normal estimator type " + self.estimate_type)

        return normals


class NormalsConv(nn.Module):
    def __init__(self, height, width, kernel_size, dilation=1, padding=0):
        super(NormalsConv, self).__init__()
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation,
                                      padding=padding, stride=1)
        # self.fold = torch.nn.Fold(output_size=(height, width), kerne_size=kernel_size,
        #                           dilation=dilation, padding=padding, stride=1)
        self.height = height
        self.width = width
        self.to3d = Depth2Points(height, width)
        if type(kernel_size) is int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        # h, w = self.kernel_size
        # self.ones = nn.Parameter(torch.ones((h * w, 1), requires_grad=False))
        # self.eye = nn.Parameter(torch.eye(3), requires_grad=False)
        # self.rays = nn.Parameter(self.to3d(torch.ones(1, 1, height, width)), requires_grad=False)

    def forward(self, depth, normals):
        _, _, height, width = depth.size()
        points = self.to3d(depth)
        rays = self.to3d(torch.ones_like(depth))
        norm_pts = torch.sum(normals * points, dim=1, keepdim=True)
        unfolded = self.unfold(torch.cat((norm_pts, rays, normals), dim=1))
        b, n, k = unfolded.size()
        Kh, Kw = self.kernel_size
        kernel_mid = Kw * (Kh // 2) + Kw // 2

        unfolded = torch.transpose(unfolded, dim0=1, dim1=2)
        unfolded = torch.reshape(unfolded, (b, k, 7, Kh * Kw))
        norms_pts_ker, rays_ker, normals_ker = torch.split(unfolded, [1, 3, 3], dim=2)
        rays_i = rays_ker[:, :, :, kernel_mid:kernel_mid + 1]
        normals_i = normals_ker[:, :, :, kernel_mid:kernel_mid + 1]

        # Calculates weights between middle normals and the rest of normals in kernel.
        # If are normals similar resulting weight will be 1.
        normal_similarity = torch.sum(normals_i * normals_ker, dim=2, keepdim=True)
        # print("minmax normal sim: ", torch.min(normal_similarity), torch.max(normal_similarity))
        norm = torch.sum(rays_i * normals_ker.detach(), dim=2, keepdim=True)
        norm = norm + (1 - torch.sign(torch.abs(norm))) * 1e-4
        corrected_depths = norms_pts_ker / norm
        similarity_norm = torch.sum(normal_similarity, dim=3) + 1e-4
        corrected_depths = torch.sum(corrected_depths * normal_similarity, dim=3) / similarity_norm
        corrected_depths = torch.reshape(corrected_depths, (b, 1, height, width))
        corrected_depths = torch.clamp(corrected_depths, min=0.0, max=8.0)
        return corrected_depths
