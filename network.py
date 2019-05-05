# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
__author__ = "Marc Eder"
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models

from util import xavier_init
from annotated_data import AnnotatedData, DataType
import numpy as np

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


class RectNet(nn.Module):

    def __init__(self, in_channels, cspn=False, reflection_pad=False,
                 normal_est=False, segmentation_est=False, plane_param_es=False,
                 calc_planes=False):
        super(RectNet, self).__init__()

        # Network definition
        self.input0_0 = ConvELUBlock(in_channels, 8, (3, 9), padding=(
            1, 4), reflection_pad=reflection_pad)
        self.input0_1 = ConvELUBlock(in_channels, 8, (5, 11), padding=(
            2, 5), reflection_pad=reflection_pad)
        self.input0_2 = ConvELUBlock(in_channels, 8, (5, 7), padding=(
            2, 3), reflection_pad=reflection_pad)
        self.input0_3 = ConvELUBlock(
            in_channels, 8, 7, padding=3, reflection_pad=reflection_pad)

        self.input1_0 = ConvELUBlock(32, 16, (3, 9), padding=(
            1, 4), reflection_pad=reflection_pad)

        self.input1_1 = ConvELUBlock(32, 16, (3, 7), padding=(
            1, 3), reflection_pad=reflection_pad)

        self.input1_2 = ConvELUBlock(32, 16, (3, 5), padding=(
            1, 2), reflection_pad=reflection_pad)

        self.input1_3 = ConvELUBlock(
            32, 16, 5, padding=2, reflection_pad=reflection_pad)

        self.encoder0_0 = ConvELUBlock(
            64, 128, 3, stride=2, padding=1, reflection_pad=reflection_pad)
        self.encoder0_1 = ConvELUBlock(
            128, 128, 3, padding=1, reflection_pad=reflection_pad)
        self.encoder0_2 = ConvELUBlock(
            128, 128, 3, padding=1, reflection_pad=reflection_pad)

        self.encoder1_0 = ConvELUBlock(
            128, 256, 3, stride=2, padding=1, reflection_pad=reflection_pad)
        self.encoder1_1 = ConvELUBlock(
            256, 256, 3, padding=2, dilation=2, reflection_pad=reflection_pad)
        self.encoder1_2 = ConvELUBlock(
            256, 256, 3, padding=4, dilation=4, reflection_pad=reflection_pad)
        self.encoder1_3 = ConvELUBlock(
            512, 256, 1, reflection_pad=reflection_pad)

        self.encoder2_0 = ConvELUBlock(
            256, 512, 3, padding=8, dilation=8, reflection_pad=reflection_pad)
        self.encoder2_1 = ConvELUBlock(
            512, 512, 3, padding=16, dilation=16, reflection_pad=reflection_pad)
        self.encoder2_2 = ConvELUBlock(
            1024, 512, 1, reflection_pad=reflection_pad)

        self.decoder0_0 = ConvTransposeELUBlock(
            512, 256, 4, stride=2, padding=1, reflection_pad=reflection_pad)

        self.decoder0_1 = ConvELUBlock(
            256, 256, 5, padding=2, reflection_pad=reflection_pad)

        self.prediction0 = nn.Conv2d(256, 1, 3, padding=1)

        self.decoder1_0 = ConvTransposeELUBlock(
            256, 128, 4, stride=2, padding=1, reflection_pad=reflection_pad)

        self.decoder1_1 = ConvELUBlock(
            128, 128, 5, padding=2, reflection_pad=reflection_pad)

        self.decoder1_2 = ConvELUBlock(
            129, 64, 1, reflection_pad=reflection_pad)

        self.prediction1 = nn.Conv2d(64, 1, 3, padding=1)

        self.cspn = cspn
        if self.cspn:
            self.guidance = ConvELUBlock(
                64, 8, 3, padding=1, reflection_pad=reflection_pad)
            self.cspn = CSPN()

        self.calc_normals = normal_est
        self.calc_plane_params = plane_param_es
        self.calc_planes = calc_planes
        if self.calc_normals or self.calc_plane_params:
            # self.decoder_norm0_0 = ConvTransposeELUBlock(
            #     512, 256, 4, stride=2, padding=1, reflection_pad=reflection_pad)

            # self.decoder_norm0_1 = ConvELUBlock(
            #     256, 256, 5, padding=2, reflection_pad=reflection_pad)

            # self.decoder_norm1_0 = ConvTransposeELUBlock(
            #     256, 128, 4, stride=2, padding=1, reflection_pad=reflection_pad)

            # self.decoder_norm1_1 = ConvELUBlock(
            #     128, 128, 5, padding=2, reflection_pad=reflection_pad)
            # self.decoder_norm1_0 = ConvTransposeELUBlock(
            #     256, 128, 4, stride=2, padding=1, reflection_pad=reflection_pad)
            # self.decoder_norm1_2 = ConvELUBlock(
            #     128, 64, 1, reflection_pad=reflection_pad)
            # self.normal_prediction = ConvELUBlock(64, 3, 3, padding=1)

            self.normal_cov1 = ConvELUBlock(64, 16, 3, padding=1)
            self.normal_cov2 = ConvELUBlock(64, 16, 3, padding=2, dilation=2)
            self.normal_cov3 = nn.Conv2d(32, 3, 1)
        self.calc_segmentation = segmentation_est
        if self.calc_segmentation:
            self.seg_cov1 = ConvELUBlock(64, 16, 3, padding=1)
            self.seg_cov2 = ConvELUBlock(64, 16, 3, padding=2, dilation=2)
            self.seg_cov3 = nn.Conv2d(32, 1, 1)

        if self.calc_planes:
            self.planar_to_depth = PlanarToDepth(height=256, width=512)
            # Initialize the network weights
        self.apply(xavier_init)

    def forward(self, inputs):
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
        upsampled_pred_2x = F.interpolate(pred_2x.detach(), scale_factor=2)

        # Second decoding block
        decoder1_0_out = self.decoder1_0(decoder0_1_out)
        # print(decoder1_0_out.size())
        # print('***********')
        decoder1_1_out = self.decoder1_1(decoder1_0_out)
        decoder1_2_out = self.decoder1_2(
            torch.cat((upsampled_pred_2x, decoder1_1_out), 1))

        # Second prediction output (original scale)
        pred_1x = self.prediction1(decoder1_2_out)

        if self.cspn:
            guidance = self.guidance(decoder1_2_out)
            opt_depth = self.cspn(
                guidance=guidance, blur_depth=pred_1x, sparse_depth=sparse_depth)
        else:
            opt_depth = pred_1x

        outputs = {}
        outputs["depth1x"] = opt_depth
        outputs["depth2x"] = pred_2x
        if self.calc_normals or self.calc_plane_params:
            normal1 = self.normal_cov1(decoder1_2_out)
            normal2 = self.normal_cov2(decoder1_2_out)
            normal_cat = torch.cat((normal1, normal2), 1)
            pred_normals = self.normal_cov3(normal_cat)

            # decoder_norm0_0_out = self.decoder_norm0_0(encoder2_2_out)
            # decoder_norm0_1_out = self.decoder_norm0_1(decoder_norm0_0_out)
            # decoder_norm1_0_out = self.decoder_norm1_0(decoder_norm0_1_out)
            # decoder_norm1_1_out = self.decoder_norm1_1(decoder_norm1_0_out)
            # decoder_norm1_2_out = self.decoder_norm1_2(decoder_norm1_1_out)
            # pred_normals = self.normal_prediction(decoder_norm1_2_out)
            # norm = torch.norm(pred_normals, p=None, keepdim=True, dim=1)
            # print(norm.size())
            if self.calc_normals:
                pred_normals = torch.tanh(pred_normals)
                outputs["normals"] = pred_normals
            else:
                outputs["PlaneParams"] = pred_normals

            # print(pred_normals.size())

        if self.calc_segmentation:
            seg1 = self.seg_cov1(decoder1_2_out)
            seg2 = self.seg_cov2(decoder1_2_out)
            seg_cat = torch.cat((seg1, seg2), 1)
            pred_seg = self.seg_cov3(seg_cat)
            outputs["segmentation"] = pred_seg

        if (self.calc_normals and self.calc_planes
                and DataType.Planes in inputs and self.calc_segmentation):
            planes = inputs[DataType.Planes]
            outputs["depth_planar"] = self.planar_to_depth(opt_depth, planes,
                                                           pred_seg, pred_normals)

        return outputs

    def annotateOutput(self, outputs):
        data = AnnotatedData()
        if "normals" in outputs:
            data.add(outputs["normals"], DataType.Normals)

        if "PlaneParams" in outputs:
            data.add(outputs["PlaneParams"], DataType.PlaneParams)

        if "segmentation" in outputs:
            data.add(outputs["segmentation"], DataType.PlanarSegmentation)
        if "depth_planar" in outputs:
            data.add(outputs["depth_planar"], DataType.Depth)
        data.add(outputs["depth1x"], DataType.Depth)
        data.add(outputs["depth2x"], DataType.Depth, scale=2)
        return data


class RectNetCSPN(nn.Module):

    def __init__(self, in_channels, cspn=False):
        super(RectNetCSPN, self).__init__()

        # Network definition
        self.input0_0 = ConvELUBlock(in_channels, 8, (3, 9), padding=(1, 4))
        self.input0_1 = ConvELUBlock(in_channels, 8, (5, 11), padding=(2, 5))
        self.input0_2 = ConvELUBlock(in_channels, 8, (5, 7), padding=(2, 3))
        self.input0_3 = ConvELUBlock(in_channels, 8, 7, padding=3)

        self.input1_0 = ConvELUBlock(32, 16, (3, 9), padding=(1, 4))
        self.input1_1 = ConvELUBlock(32, 16, (3, 7), padding=(1, 3))
        self.input1_2 = ConvELUBlock(32, 16, (3, 5), padding=(1, 2))
        self.input1_3 = ConvELUBlock(32, 16, 5, padding=2)

        self.encoder0_0 = ConvELUBlock(64, 128, 3, stride=2, padding=1)
        self.encoder0_1 = ConvELUBlock(128, 128, 3, padding=1)
        self.encoder0_2 = ConvELUBlock(128, 128, 3, padding=1)

        self.encoder1_0 = ConvELUBlock(128, 256, 3, stride=2, padding=1)
        self.encoder1_1 = ConvELUBlock(256, 256, 3, padding=2, dilation=2)
        self.encoder1_2 = ConvELUBlock(256, 256, 3, padding=4, dilation=4)
        self.encoder1_3 = ConvELUBlock(512, 256, 1)

        self.encoder2_0 = ConvELUBlock(256, 512, 3, padding=8, dilation=8)
        self.encoder2_1 = ConvELUBlock(512, 512, 3, padding=16, dilation=16)
        self.encoder2_2 = ConvELUBlock(1024, 512, 1)

        self.decoder0_0 = ConvTransposeELUBlock(
            512, 256, 4, stride=2, padding=1)
        self.decoder0_1 = ConvELUBlock(256, 256, 5, padding=2)

        self.prediction0 = nn.Conv2d(256, 1, 3, padding=1)

        self.decoder1_0 = ConvTransposeELUBlock(
            256, 128, 4, stride=2, padding=1)
        self.decoder1_1 = ConvELUBlock(128, 128, 5, padding=2)
        self.decoder1_2 = ConvELUBlock(129, 64, 1)

        self.prediction1 = nn.Conv2d(64, 1, 3, padding=1)

        self.cspn = True
        if self.cspn:
            self.guid_decode = ConvTransposeELUBlock(
                256, 128, 4, stride=2, padding=1)
            self.guid_conv_1 = ConvELUBlock(128, 128, 5, padding=2)
            # self.guid_conv_2 = ConvELUBlock(129, 64, 1)
            self.guid_conv_3 = ConvELUBlock(128, 8, 3, padding=1)
            self.cspn = CSPN()
        # Initialize the network weights
        self.apply(xavier_init)

    def forward(self, inputs):
        x = inputs[DataType.Image]
        if DataType.SparseDepth in inputs:
            sparse_depth = inputs[DataType.SparseDepth][1]
            x = torch.cat((x, sparse_depth), 1)
        else:
            sparse_depth = None

        # First filter bank
        input0_0_out = self.input0_0(x)
        input0_1_out = self.input0_1(x)
        input0_2_out = self.input0_2(x)
        input0_3_out = self.input0_3(x)
        input0_out_cat = torch.cat(
            (input0_0_out,
             input0_1_out,
             input0_2_out,
             input0_3_out), 1)

        # Second filter bank
        input1_0_out = self.input1_0(input0_out_cat)
        input1_1_out = self.input1_1(input0_out_cat)
        input1_2_out = self.input1_2(input0_out_cat)
        input1_3_out = self.input1_3(input0_out_cat)

        # First encoding block
        encoder0_0_out = self.encoder0_0(
            torch.cat((input1_0_out,
                       input1_1_out,
                       input1_2_out,
                       input1_3_out), 1))
        encoder0_1_out = self.encoder0_1(encoder0_0_out)
        encoder0_2_out = self.encoder0_2(encoder0_1_out)

        # Second encoding block
        encoder1_0_out = self.encoder1_0(encoder0_2_out)
        encoder1_1_out = self.encoder1_1(encoder1_0_out)
        encoder1_2_out = self.encoder1_2(encoder1_1_out)
        encoder1_3_out = self.encoder1_3(
            torch.cat((encoder1_1_out, encoder1_2_out), 1))

        # Third encoding block
        encoder2_0_out = self.encoder2_0(encoder1_3_out)
        encoder2_1_out = self.encoder2_1(encoder2_0_out)
        encoder2_2_out = self.encoder2_2(
            torch.cat((encoder2_0_out, encoder2_1_out), 1))

        # First decoding block
        decoder0_0_out = self.decoder0_0(encoder2_2_out)
        decoder0_1_out = self.decoder0_1(decoder0_0_out)

        # 2x downsampled prediction
        pred_2x = self.prediction0(decoder0_1_out)
        upsampled_pred_2x = F.interpolate(pred_2x.detach(), scale_factor=2)

        # Second decoding block
        decoder1_0_out = self.decoder1_0(decoder0_1_out)
        decoder1_1_out = self.decoder1_1(decoder1_0_out)
        decoder1_2_out = self.decoder1_2(
            torch.cat((upsampled_pred_2x, decoder1_1_out), 1))

        # Second prediction output (original scale)
        pred_1x = self.prediction1(decoder1_2_out)

        if self.cspn:
            guidance0 = self.guid_decode(decoder0_1_out)
            guidance1 = self.guid_conv_1(guidance0)
            # guidance2 = self.guid_conv_2(guidance1)
            opt_depth = self.cspn(
                guidance=guidance1, blur_depth=pred_1x, sparse_depth=sparse_depth)
            return [opt_depth, pred_1x, pred_2x]
        else:
            return [pred_1x, pred_2x]

    def annotateOutput(self, outputs):
        data = AnnotatedData()
        data.add(outputs[0], DataType.Depth)

        if self.cspn:
            data.add(outputs[1], DataType.Depth, scale=1)
            data.add(outputs[2], DataType.Depth, scale=2)
        else:
            data.add(outputs[1], DataType.Depth, scale=2)
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
                 reflection_pad=False):
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

    def forward(self, x):
        if self.reflection_pad:
            return F.elu(self.conv(self.padding_l(x)), inplace=True)
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


class Depth2Points(nn.Module):
    def __init__(self, height, width):
        super(Depth2Points, self).__init__()
        phi = torch.zeros((height, width))
        theta = torch.zeros((height, width))

        hcam_deg = 360
        vcam_deg = 180
        # Camera rotation angles in radians
        hcam_rad = hcam_deg / 180.0 * np.pi
        vcam_rad = vcam_deg / 180.0 * np.pi
        # print(hcam_deg, vcam_deg)
        for v in range(height):
            for u in range(width):
                theta[v, u] = (u - width / 2.0) / width * hcam_rad
                phi[v, u] = -(v - height / 2.0) / height * vcam_rad
        self.cos_theta = nn.Parameter(torch.cos(theta), requires_grad=False)
        self.sin_theta = nn.Parameter(torch.sin(theta), requires_grad=False)
        self.cos_phi = nn.Parameter(torch.cos(phi), requires_grad=False)
        self.sin_phi = nn.Parameter(torch.sin(phi), requires_grad=False)

    def forward(self, depth):
        X = depth * self.cos_phi * self.cos_theta
        Y = depth * self.cos_phi * self.sin_theta
        Z = depth * self.sin_phi
        points = torch.cat((X, Y, Z), dim=1)
        # print(points)
        return points


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
    # print(params.size(), planes.size(), planes3_channel.size())
    avg_param = torch.sum(params * planes3_channel, dim=3)
    # print(avg_param.size(), plane_weights.size())
    # b x p x 3
    avg_param /= plane_weights
    avg_param = torch.reshape(avg_param, (*avg_param.size(), 1))
    if keepdim:
        return avg_param * planes3_channel
    else:
        return avg_param


class PlanarToDepth(nn.Module):
    def __init__(self, height, width):
        super(PlanarToDepth, self).__init__()
        self.to3d = Depth2Points(height, width)

    def forward(self, depth, planes, segmentation, normals):
        points = self.to3d(depth)
        rays = self.to3d(torch.ones_like(depth))
        b, p, _, h, w = planes.size()
        planes = torch.reshape(planes, (b, p, 1, -1))
        points = torch.reshape(points, (b, 1, 3, -1))
        depths = torch.reshape(depth, (b, 1, 1, -1))
        rays = torch.reshape(rays, (b, 1, 3, -1))
        # BxPx3xN
        avg_normals = planeAwarePooling(normals, planes, keepdim=True)
        centroids = planeAwarePooling(depths, planes, keepdim=True)

        norm = torch.sum(rays * avg_normals, dim=2) + 0.000001
        planar_depths = (torch.sum(centroids * avg_normals, dim=2) / norm)
        # B x P x N
        planar_depths = torch.sum(planar_depths, dim=1)
        planar_depths = torch.reshape(planar_depths, (b, 1, h, w))
        return segmentation * planar_depths + (1 - segmentation) * depth


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
