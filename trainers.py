import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import math
import shutil
import os.path as osp
import os

import util
from metrics import (abs_rel_error, delta_inlier_ratio,
                     lin_rms_sq_error, log_rms_sq_error,
                     sq_rel_error, directed_depth_error,
                     depth_boundary_error, planarity_errors,
                     delta_normal_angle_ratio, normal_stats)

from util import (saveTensorDepth, toDevice, register_hooks, plot_grad_flow, imageHeatmap,
                  stackVerticaly, heatmapGrid, pytorchDetachedProcess, linePlot)
from network import toPlaneParams, DepthToNormals
import torchvision.transforms as Tt
from torchvision.utils import make_grid

from annotated_data import AnnotatedData, DataType
import colorsys
import numpy as np
import json
import plotly as py
from visualizations import panoDepthToPcl, savePcl
# From https://github.com/fyu/drn


class SeriesData(object):
    def __init__(self):
        self.reset()

    def update(self, x, y, std=0):
        self.x.append(x)
        self.y.append(y)
        self.std.append(std)
        self.steps += 1

    def reset(self):
        self.steps = 0
        self.x = []
        self.y = []
        self.std = []


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.elems = []

    def update(self, val, n=1):
        # print(val)
        if isinstance(val, (torch.Tensor)):
            val = val.cpu().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.elems.append(val)
        # print(self.elems)
        self.std = np.std(self.elems)

    def to_dict(self):
        return {'val': self.val,
                'sum': self.sum,
                'count': self.count,
                'avg': self.avg,
                'std': self.std,
                'elems': json.dumps(self.elems)
                }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']
        if 'std' in meter_dict:
            self.elems = json.loads(meter_dict["elems"])
            self.std = meter_dict["std"]


def resetAverageMeters(average_meters):
    for key, val in average_meters.items():
        val.reset()


def loadAverageMeters(average_meters):
    loaded_meters = {}
    for key, val in average_meters.items():
        loaded_meters[key] = AverageMeter()
        loaded_meters[key].from_dict(val)
    return loaded_meters


def serializeAverageMeters(average_meters):
    final_dict = {}
    for key, val in average_meters.items():
        final_dict[key] = val.to_dict()
    return final_dict


def visualize_rgb(rgb):
    # Scale back to [0,255]
    return (255 * rgb).byte()


def visualize_mask(mask):
    '''Visualize the data mask'''
    mask /= mask.max()
    return (255 * mask).byte()


def visualizePlanes(planes):
    p, h, w = planes.size()
    HSV_tuples = [(i / 360.0,
                   np.random.uniform(0.8, 0.9),
                   np.random.uniform(0.7, 0.8))
                  for i in np.arange(0, 360.0, 360.0 / p)]
    RGB_tuples = list(map(lambda x: colorsys.hls_to_rgb(*x), HSV_tuples))

    R = torch.zeros((1, h, w))
    G = torch.zeros((1, h, w))
    B = torch.zeros((1, h, w))

    for i, (r, g, b) in enumerate(RGB_tuples):
        R += planes[i] * r
        G += planes[i] * g
        B += planes[i] * b

    img = torch.cat((R, G, B), dim=0)
    img[img == 0] = 1
    return img


def parse_data(raw_data):
    '''
        Data must be instace of AnnotatationData
    '''
    # print(data)
    #  convert to anotated form. This is required because DataLoader expect only basic types
    data = AnnotatedData()
    for key, tensor in raw_data[0].items():
        data.add(tensor, data_type=key, scale=1)

    data.filenames = raw_data[1]

    rgb = data.get(DataType.Image, scale=1)[0]
    gt_depth_1x = data.get(DataType.Depth, scale=1)
    if len(gt_depth_1x):
        gt_depth_1x = gt_depth_1x[0]
        gt_depth_2x = F.interpolate(gt_depth_1x, scale_factor=0.5)
        gt_depth_4x = F.interpolate(gt_depth_1x, scale_factor=0.25)
        data.add(gt_depth_2x, DataType.Depth, scale=2)
        data.add(gt_depth_4x, DataType.Depth, scale=4)

    mask_1x = data.get(DataType.Mask, scale=1)
    if len(mask_1x):
        mask_1x = mask_1x[0]
        mask_2x = F.interpolate(mask_1x, scale_factor=0.5)
        mask_4x = F.interpolate(mask_1x, scale_factor=0.25)
        data.add(mask_2x, DataType.Mask, scale=2)
        data.add(mask_4x, DataType.Mask, scale=4)

    inputs = {DataType.Image: rgb}
    planes = data.get(DataType.Planes, scale=1)
    if len(planes):
        inputs[DataType.Planes] = planes[0]

    return inputs, data


def genSparsePontsParser(empty_sparse_pts=False):
    def parse_data_sparse_depth(raw_data):
        '''
        Returns a list of the inputs as first output, a list of the GT as a second output,
        and a list of the remaining info as a third output. Must be implemented.
        '''
        inputs, data = parse_data(raw_data)

        sparse_depth_1x = data.get(DataType.SparseDepth, scale=1)[0]
        sparse_depth_2x = F.interpolate(sparse_depth_1x, scale_factor=0.5)
        sparse_depth_4x = F.interpolate(sparse_depth_1x, scale_factor=0.25)

        sparse_scales = {1: sparse_depth_1x, 2: sparse_depth_2x, 4: sparse_depth_4x}
        if empty_sparse_pts:
            sparse_scales = {
                1: torch.zeros_like(sparse_depth_1x),
                2: torch.zeros_like(sparse_depth_2x),
                4: torch.zeros_like(sparse_depth_4x)
            }
        inputs[DataType.SparseDepth] = sparse_scales

        data.add(sparse_scales[1], DataType.SparseDepth, scale=1)
        data.add(sparse_scales[2], DataType.SparseDepth, scale=2)
        data.add(sparse_scales[4], DataType.SparseDepth, scale=4)

        return inputs, data
    return parse_data_sparse_depth


def save_samples_default(data, outputs, results_dir):
    '''
    Saves samples of the network inputs and outputs
    '''
    pass


def save_saples_for_pcl(data, outputs, results_dir):
    # print(outputs)
    d = data
    o = outputs.get(DataType.Depth, scale=1)[0].cpu()

    batch_size, _, _, _ = o.size()
    # print(batch_size)
    for i in range(batch_size):
        # print(i)
        unnorm_depth = 8
        name = d.filenames[i]
        # print(name)
        name = osp.join(results_dir, name)
        prediction = o[i]
        img = d.get(DataType.Image, scale=1)[0][i]

        gt = d.get(DataType.Depth, scale=1)
        if len(gt):
            saveTensorDepth(name + "_gt_depth", gt[0][i], 1)
        saveTensorDepth(name + "_pred_depth", prediction, 1)

        sparse_points = d.get(DataType.SparseDepth, scale=1)
        if len(sparse_points) > 0:
            sparse_points = sparse_points[0][i]
            saveTensorDepth(name + "_sparse_depth",
                            sparse_points, unnorm_depth)

        color_img = Tt.functional.to_pil_image(img.cpu())
        color_img.save(name + "_color.jpg")

        # d['original_image'][i].write(osp.join(results_dir, name + "_color.jpg"))


def genTotalLoss(factors={}):
    def totalLoss(loss):
        for key, factor in factors.items():
            if key in loss:
                loss[key] *= factor
            else:
                print("multiplyTotalLoss: missing loss: ", key, "should be multiplied by ", factor)

        total_loss = 0
        if isinstance(loss, dict):
            for key, val in loss.items():
                total_loss += val
        elif isinstance(loss, list):
            for val in loss.items():
                total_loss += val
        elif isinstance(loss, torch.Tensor):
            assert(not torch.isnan(loss).any())
            total_loss = loss
        else:
            raise ValueError("Unsupoted loss type: ", type(loss))
        return total_loss
    return totalLoss


class MonoTrainer(object):

    def __init__(
            self,
            name,
            network,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer,
            checkpoint_dir,
            device,
            parse_fce=parse_data,
            save_samples_fce=save_samples_default,
            sum_losses_fce=None,
            visdom=None,
            scheduler=None,
            num_epochs=20,
            validation_freq=1,
            visualization_freq=5,
            validation_sample_freq=-1,
            batch_checkoint_freq=100):

        # Name of this experiment
        self.name = name
        self.parse_data = parse_fce
        self.save_samples = save_samples_fce
        if sum_losses_fce is None:
            self.sumLosses = genTotalLoss()
        else:
            self.sumLosses = sum_losses_fce

        # Class instances
        self.network = network
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training options
        self.num_epochs = num_epochs
        self.validation_freq = validation_freq
        self.visualization_freq = visualization_freq
        self.validation_sample_freq = validation_sample_freq
        self.batch_checkoint_freq = batch_checkoint_freq

        # CUDA info
        self.device = device

        # Some timers
        self.batch_time_meter = AverageMeter()
        self.forward_time_meter = AverageMeter()
        self.backward_time_meter = AverageMeter()
        self.data_prep_time_meter = AverageMeter()
        self.loss_time_meter = AverageMeter()

        # Some trackers
        self.epoch = 0

        # Directory to store checkpoints
        self.checkpoint_dir = checkpoint_dir

        # Accuracy metric trackers
        self.err_meters = {
            'Abs. Rel. Error': AverageMeter(),
            'Sq. Rel. Error': AverageMeter(),
            'Linear RMS Error': AverageMeter(),
            'Log RMS Error': AverageMeter(),
            "Sparse Pts. Abs Rel": AverageMeter(),
            "Direct Depth M Err": AverageMeter(),
            "Direct Depth P Err": AverageMeter(),
            "Direct Depth 0 Err": AverageMeter(),
            "Depth Boundary Err": AverageMeter(),
            "Normal Mean": AverageMeter(),
            "Normal Median": AverageMeter(),
            "Plane Flatness Err": AverageMeter(),
            "Plane Orientation Err": AverageMeter(),
        }

        self.acc_meters = {
            "Inlier D1": AverageMeter(),
            "Inlier D2": AverageMeter(),
            "Inlier D3": AverageMeter(),
            "Depth Boundary Acc": AverageMeter(),
            "Normal Angle 11.25": AverageMeter(),
            "Normal Angle 22.5": AverageMeter(),
            "Normal Angle 30": AverageMeter(),
        }
        # over multiple steps tracker
        self.err_trackers = {
            'Abs. Rel. Error': SeriesData(),
            'Sq. Rel. Error': SeriesData(),
            'Linear RMS Error': SeriesData(),
            'Log RMS Error': SeriesData(),
            "Sparse Pts. Abs Rel": SeriesData(),
            "Direct Depth M Err": SeriesData(),
            "Direct Depth P Err": SeriesData(),
            "Direct Depth 0 Err": SeriesData(),
            "Depth Boundary Err": SeriesData(),
            "Normal Mean": SeriesData(),
            "Normal Median": SeriesData(),
            "Plane Flatness Err": SeriesData(),
            "Plane Orientation Err": SeriesData(),
        }

        self.acc_trackers = {
            "Inlier D1": SeriesData(),
            "Inlier D2": SeriesData(),
            "Inlier D3": SeriesData(),
            "Depth Boundary Acc": SeriesData(),
            "Normal Angle 11.25": SeriesData(),
            "Normal Angle 22.5": SeriesData(),
            "Normal Angle 30": SeriesData(),
        }

        # Track the best inlier ratio recorded so far
        self.best_d1_inlier = 0.0

        # List of length 2 [Visdom instance, env]
        self.vis = visdom

        # Loss trackers
        self.loss_meters = {
            "total": AverageMeter()
        }
        self.loss_trackers = {
            "total": SeriesData()
        }
        self.dry_run = False
        self.save_to_disk = True

    def setDryRun(self, state):
        self.dry_run = state

    def forward_pass(self, inputs):
        '''
        Accepts the inputs to the network. The input should be  \
            a list or dictionary of tensors
        Returns the network output
        '''
        return self.network(inputs)

    def annotate_outputs(self, outputs):
        if hasattr(self.network, 'annotateOutput'):
            return self.network.annotateOutput(outputs)

        if hasattr(self.network, 'module'):
            try:
                return self.network.module.annotateOutput(outputs)
            except AttributeError:
                raise Exception("Implement method annotateOutput in " + repr(self.network.model))
        raise Exception("Implement method annotateOutput in " + repr(self.network))

    def compute_loss(self, output, data):
        '''
        Returns the total loss
        '''
        return self.criterion(output, data)

    def backward_pass(self, loss):
        # Computes the backward pass and updates the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def updateLossMeters(self, raw_loss, total_loss):
        self.loss_meters["total"].update(total_loss.item())
        if isinstance(raw_loss, dict):
            for key, val in raw_loss.items():
                if key not in self.loss_meters:
                    self.loss_meters[key] = AverageMeter()
                    self.loss_trackers[key] = SeriesData()
                self.loss_meters[key].update(val.item())

    def train_one_epoch(self):

        # Put the model in train mode
        self.network = self.network.train()
        resetAverageMeters(self.loss_meters)
        # Load data
        end = time.time()
        for batch_num, raw_data in enumerate(self.train_dataloader):
            # Parse the data into inputs, ground truth, and other
            # torch.cuda.synchronize()
            data_prep_time = time.time()
            inputs, data = self.parse_data(raw_data)
            inputs = toDevice(inputs, self.device)
            data = data.to(self.device)
            # torch.cuda.synchronize()
            self.data_prep_time_meter.update(time.time() - data_prep_time)
            # print(data.get(DataType.Normals, scale=1)[0].size())
            if self.dry_run:
                continue
            # Run a forward pass
            # torch.cuda.synchronize()
            forward_time = time.time()
            output = self.annotate_outputs(self.forward_pass(inputs))
            # torch.cuda.synchronize()
            self.forward_time_meter.update(time.time() - forward_time)

            # torch.cuda.synchronize()
            loss_time = time.time()
            # Compute the loss(es)
            raw_loss, histograms = self.compute_loss(output, data)
            loss = self.sumLosses(raw_loss)
            # torch.cuda.synchronize()
            self.loss_time_meter.update(time.time() - loss_time)

            # Backpropagation of the total loss
            # torch.cuda.synchronize()
            backward_time = time.time()
            self.backward_pass(loss)
            # torch.cuda.synchronize()
            self.backward_time_meter.update(time.time() - backward_time)

            # Update batch times
            self.batch_time_meter.update(time.time() - end)
            end = time.time()
            self.updateLossMeters(raw_loss, loss)

            if batch_num % self.batch_checkoint_freq == 0:
                self.save_checkpoint(is_best=False)
            # Every few batches
            if batch_num % self.visualization_freq == 0:
                dict_loss = {}
                for key, tensor in raw_loss.items():
                    dict_loss[key] = tensor.item()
                print(json.dumps(dict_loss, indent=4, sort_keys=True))
                # Visualize the loss
                total_batches = self.epoch * len(self.train_dataloader) + batch_num

                for key, loss_meter in self.loss_meters.items():
                    self.loss_trackers[key].update(total_batches, loss_meter.avg, loss_meter.std)

                visualize_loss(self.vis, self.loss_trackers, self.checkpoint_dir, False)
                visualize_samples(self.vis, self.checkpoint_dir, data, output, histograms,
                                  self.save_to_disk)

                # Print the most recent batch report
                self.print_batch_report(batch_num)
                resetAverageMeters(self.loss_meters)

    def predict(self, checkpoint_path):
        self.validate(checkpoint_path, save_all_predictions=True, calc_metrics=False)

    def validate(self, checkpoint_path=None, save_all_predictions=False, calc_metrics=True):
        print('Validating model....')

        if checkpoint_path is not None:
            self.load_checkpoint(
                checkpoint_path, weights_only=True, eval_mode=True)

        # Put the model in eval mode
        self.network = self.network.eval()

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, raw_data in tqdm(enumerate(self.val_dataloader),
                                            total=len(self.val_dataloader)):

                # Parse the data
                inputs, data = self.parse_data(raw_data)
                data = data.to(self.device)
                inputs = toDevice(inputs, self.device)

                # Run a forward pass
                output = self.annotate_outputs(self.forward_pass(inputs))

                # Compute the evaluation metrics
                if calc_metrics:
                    self.compute_eval_metrics(output, data)
                if save_all_predictions:
                    self.save_samples(data, output, self.checkpoint_dir)
                else:
                    # If trying to save intermediate outputs
                    if self.validation_sample_freq >= 0:
                        # Save the intermediate outputs
                        if batch_num % self.validation_sample_freq == 0:
                            self.save_samples(
                                data, output, self.checkpoint_dir)

        # Print a report on the validation results
        print('Validation finished in {} seconds'.format(time.time() - s))
        report = self.print_validation_report()
        self.network = self.network.train()
        return report

    def train(self, checkpoint_path=None, weights_only=False):

        # Load pretrained parameters if desired
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, weights_only)
            # if weights_only:
            # self.initialize_visualizations()
        # else:
            # Initialize any training visualizations
            # self.initialize_visualizations()
            # make sure that validation is correct without bugs

        self.validate()
        self.visualize_metrics()
        print('Starting training')
        # Train for specified number of epochs
        for self.epoch in range(self.epoch, self.num_epochs):

            # Increment the LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            self.network = self.network.train()
            # Run an epoch of training
            self.train_one_epoch()

            if self.epoch % self.validation_freq == 0:
                is_best = False
                self.validate()
                # Also update the best state tracker
                if self.best_d1_inlier < self.d1_inlier_meter.avg:
                    self.best_d1_inlier = self.d1_inlier_meter.avg
                    is_best = True
                self.save_checkpoint(is_best)
                self.visualize_metrics()

    def reset_eval_metrics(self):
        '''
        Resets metrics used to evaluate the model
        '''
        for key, metric in self.err_meters.items():
            metric.reset()
        for key, metric in self.acc_meters.items():
            metric.reset()

    def compute_eval_metrics(self, output, data):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output.get(DataType.Depth, scale=1)[0]

        gt_depth = data.get(DataType.Depth, scale=1)[0]
        depth_mask = data.get(DataType.Depth, scale=1)[0]
        sparse_pts = data.get(DataType.SparseDepth, scale=1)
        gt_normals = data.get(DataType.Normals, scale=1)
        pred_normals = output.get(DataType.Normals, scale=1)
        gt_planes = data.get(DataType.Planes)
        if len(sparse_pts) > 0:
            pts_present_mask = torch.sign(sparse_pts[0])
            depth_mask = depth_mask - depth_mask * pts_present_mask

        N = depth_mask.sum().item()

        # Align the prediction scales via median
        median_scaling_factor = gt_depth[depth_mask > 0].median(
        ) / depth_pred[depth_mask > 0].median()
        depth_pred *= median_scaling_factor

        abs_rel = abs_rel_error(depth_pred, gt_depth, depth_mask)
        sq_rel = sq_rel_error(depth_pred, gt_depth, depth_mask)
        rms_sq_lin = lin_rms_sq_error(depth_pred, gt_depth, depth_mask)
        rms_sq_log = log_rms_sq_error(depth_pred, gt_depth, depth_mask)
        d1 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=1)
        d2 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=2)
        d3 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=3)
        dde_0, dde_m, dde_p = directed_depth_error(depth_pred, gt_depth, depth_mask, thr=3.0)
        dbe_acc, dbe_com = depth_boundary_error(depth_pred, gt_depth, depth_mask)

        # if len(gt_normals) > 0 and len(gt_planes) > 0:
        #     pe_flat, orient_err = planarity_errors(depth_pred, gt_normals[0],
        #                                            gt_planes[0], depth_mask)

        if len(gt_normals) > 0 and len(pred_normals) > 0:
            norm1 = delta_normal_angle_ratio(pred_normals[0], gt_normals[0],
                                             depth_mask, degree=11.25)
            norm2 = delta_normal_angle_ratio(pred_normals[0], gt_normals[0],
                                             depth_mask, degree=22.5)
            norm3 = delta_normal_angle_ratio(pred_normals[0], gt_normals[0],
                                             depth_mask, degree=30)
            norm_mean, norm_median = normal_stats(pred_normals[0], gt_normals[0], depth_mask)
        # pe_flat, orient_err = planarity_errors(depth_pred, gt_normals, planes, mask)
        if len(sparse_pts) > 0:
            sparse_abs_rel = abs_rel_error(depth_pred, gt_depth, pts_present_mask)

        self.err_meters["Abs. Rel. Error"].update(abs_rel, N)
        self.err_meters["Sq. Rel. Error"].update(sq_rel, N)
        self.err_meters["Linear RMS Error"].update(rms_sq_lin, N)
        self.err_meters["Log RMS Error"].update(rms_sq_log, N)
        self.err_meters["Direct Depth M Err"].update(dde_m)
        self.err_meters["Direct Depth P Err"].update(dde_p)
        self.err_meters["Direct Depth 0 Err"].update(dde_0)
        self.err_meters["Depth Boundary Err"].update(dbe_com)

        if len(sparse_pts) > 0:
            self.err_meters["Sparse Pts. Abs Rel"].update(sparse_abs_rel.item(),
                                                          pts_present_mask.sum().item())

        self.acc_meters["Inlier D1"].update(d1, N)
        self.acc_meters["Inlier D2"].update(d2, N)
        self.acc_meters["Inlier D3"].update(d3, N)
        self.acc_meters["Depth Boundary Acc"].update(dbe_acc)

        if len(gt_normals) > 0 and len(pred_normals) > 0:
            self.acc_meters["Normal Angle 11.25"].update(norm1)
            self.acc_meters["Normal Angle 22.5"].update(norm2)
            self.acc_meters["Normal Angle 30"].update(norm3)
            self.err_meters["Normal Mean"].update(norm_mean)
            self.err_meters["Normal Median"].update(norm_median)

        # if len(gt_normals) > 0 and len(gt_planes) > 0:
        #     self.err_meters["Plane Flatness Err"].update(pe_flat)
        #     self.err_meters["Plane Orientation Err"].update(orient_err)

    def load_checkpoint(self, checkpoint_path=None, weights_only=False, eval_mode=False):
        '''
        Initializes network with pretrained parameters
        '''
        if checkpoint_path is not None:
            print('Loading checkpoint \'{}\''.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

            # If we want to continue training where we left off, load entire training state
            if weights_only:
                print('NOTE: Loading weights only')
            else:
                self.epoch = checkpoint['epoch']
                experiment_name = checkpoint['experiment']
                self.vis[1] = experiment_name
                self.best_d1_inlier = checkpoint['best_d1_inlier']
                if "loss_meter" in checkpoint:
                    self.loss_meters["total"].from_dict(checkpoint["loss_meter"])
                if "loss_meters" in checkpoint:
                    self.loss_meters = loadAverageMeters(checkpoint["loss_meters"])
                # Load the optimizer
                if not eval_mode:
                    util.load_optimizer(
                        self.optimizer,
                        checkpoint['optimizer'],
                        self.device)
            # Load model
            util.load_partial_model(
                self.network,
                checkpoint['state_dict'])

            print('Loaded checkpoint \'{}\' (epoch {})'.format(
                checkpoint_path, checkpoint['epoch']))
        else:
            print('WARNING: No checkpoint found')

    def visualize_metrics(self):
        '''
        Updates the metrics visualization
        '''
        epoch = self.epoch + 1
        for key, meter in self.acc_meters.items():
            self.acc_trackers[key].update(epoch, meter.avg, meter.std)

        for key, meter in self.err_meters.items():
            self.err_trackers[key].update(epoch, meter.avg, meter.std)

        res_folder = osp.join(self.checkpoint_dir, "visdom")
        traces = []
        for key, tracker in self.err_trackers.items():
            traces.append((key, tracker.x, tracker.y, tracker.std))
        graph = linePlot(traces)
        py.io.write_json(graph, osp.join(res_folder, "errors.json"))
        # with open(osp.join(res_folder, "errors.json"), mode="w") as f:
        #     f.write(json.dumps(graph))

        if self.save_to_disk:
            py.io.write_image(graph, osp.join(res_folder, "errors.png"))
        if self.vis is not None:
            self.vis[0].plotlyplot(graph, win="error_metrics", env=self.vis[1])

        traces = []
        for key, tracker in self.acc_trackers.items():
            traces.append((key, tracker.x, tracker.y, tracker.std))
        graph = linePlot(traces)
        py.io.write_json(graph, osp.join(res_folder, "acc.json"))
        # with open(osp.join(res_folder, "acc.json"), mode="w") as f:
        #     f.write(json.dumps(graph))

        if self.save_to_disk:
            py.io.write_image(graph, osp.join(res_folder, "acc.png"))
        if self.vis is not None:
            self.vis[0].plotlyplot(graph, win="inlier_metrics", env=self.vis[1])

    def print_batch_report(self, batch_num):
        '''
        Prints a report of the current batch
        '''
        print('Epoch: [{0}][{1}/{2}]\t'
              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
              'Data prep Time {data_prep_time.val:.3f} ({data_prep_time.avg:.3f})\t'
              'Forward Time {forward_time.val:.3f} ({forward_time.avg:.3f})\n'
              'Loss Time {loss_time.val:.3f} ({loss_time.avg:.3f})\t'
              'Backward Time {backward_time.val:.3f} ({backward_time.avg:.3f})\n'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\n\n'.format(
                  self.epoch + 1,
                  batch_num + 1,
                  len(self.train_dataloader),
                  batch_time=self.batch_time_meter,
                  forward_time=self.forward_time_meter,
                  backward_time=self.backward_time_meter,
                  data_prep_time=self.data_prep_time_meter,
                  loss_time=self.loss_time_meter,
                  loss=self.loss_meters["total"]))

    def print_validation_report(self):
        '''
        Prints a report of the validation results
        '''
        lines = []
        lines.append("Epoch: {}\n".format(self.epoch + 1))
        for key, meter in self.err_meters.items():
            lines.append("{}: {:.4f} ({:.4f})\n".format(key, meter.avg, meter.std))

        for key, meter in self.acc_meters.items():
            lines.append("{}: {:.4f} ({:.4f})\n".format(key, meter.avg, meter.std))

        report = ''.join(lines)
        print(report)
        return report

    def save_checkpoint(self, is_best):
        '''
        Saves the model state
        '''
        # Save latest checkpoint (constantly overwriting itself)
        checkpoint_path = osp.join(
            self.checkpoint_dir,
            'checkpoint_latest.pth')

        # avoid exporting DataParallel
        if isinstance(self.network, torch.nn.DataParallel):
            state_dict = self.network.module.state_dict()
        else:
            state_dict = self.network.state_dict()

        # Actually saves the latest checkpoint and also updating the file holding the best one
        util.save_checkpoint(
            {
                'epoch': self.epoch,
                'experiment': self.name,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'loss_meters': serializeAverageMeters(self.loss_meters),
                'best_d1_inlier': self.best_d1_inlier
            },
            is_best,
            filename=checkpoint_path)

        # Copies the latest checkpoint to another file stored for each epoch
        history_path = osp.join(
            self.checkpoint_dir,
            'checkpoint_{:03d}.pth'.format(self.epoch + 1))
        shutil.copyfile(checkpoint_path, history_path)
        print('Checkpoint saved')


@pytorchDetachedProcess
def visualize_samples(visdom, directory, data, output, loss_hist, save_to_disk=True):
    '''
    Updates the output samples visualization
    '''

    data_folder = osp.join(directory, "visdom")
    os.makedirs(data_folder, exist_ok=True)
    # visdom[0].save([visdom[1]])
    rgb = data.get(DataType.Image, scale=1)[0][0, :, :, :].cpu().detach()
    depth_mask = data.get(DataType.Mask, scale=1)[0][0].cpu().detach()
    pred_depth = output.get(DataType.Depth, scale=1)[0][0].cpu().detach()
    pred_depth *= depth_mask
    pred_depth = pred_depth.squeeze().flip(0).numpy()

    gt_depth = data.get(DataType.Depth, scale=1)[0][0]
    gt_depth = gt_depth.cpu().detach()
    gt_depth *= depth_mask
    gt_depth = gt_depth.squeeze().flip(0).numpy()

    for key, hist in loss_hist.items():
        # print(key)
        max_val = torch.max(hist[0]).item()
        if max_val > 8:
            max_val = 8.0
        # print(key)
        graph = imageHeatmap(rgb, hist[0].cpu().squeeze().flip(0), title=key, max_val=max_val)
        file = osp.join(data_folder, key + '_hist.json')
        py.io.write_json(graph, file)

        if save_to_disk:
            py.io.write_image(graph, osp.join(data_folder, key + '_hist.png'))
        if visdom is not None:
            visdom[0].plotlyplot(graph, win=key, env=visdom[1])

    for i, (scale, guidance) in enumerate(output.queryType(DataType.Guidance)):
        key = "guidance_" + str(i)
        graph = imageHeatmap(rgb, guidance[0].cpu().squeeze().flip(0), title=key, max_val=None)
        file = osp.join(data_folder, key + '.json')
        py.io.write_json(graph, file)

        if save_to_disk:
            py.io.write_image(graph, osp.join(data_folder, key + '.png'))
        if visdom is not None:
            visdom[0].plotlyplot(graph, win=key, env=visdom[1])

    if visdom is not None:
        visdom[0].image(
            visualize_rgb(rgb),
            env=visdom[1],
            win='rgb',
            opts=dict(
                title='Input RGB Image',
                caption='Input RGB Image'))

        visdom[0].image(
            visualize_mask(depth_mask),
            env=visdom[1],
            win='mask',
            opts=dict(
                title='Mask',
                caption='Mask'))

    depth_fig = heatmapGrid([gt_depth, pred_depth], ["Gt depth", "Pred depth"], columns=1)
    file = osp.join(data_folder, 'depths.json')
    py.io.write_json(depth_fig, file)

    if save_to_disk:
        py.io.write_image(depth_fig, osp.join(data_folder, 'depths.png'))
    if visdom is not None:
        visdom[0].plotlyplot(depth_fig, win="depths", env=visdom[1])

    pred_seg = output.get(DataType.PlanarSegmentation, scale=1)
    gt_seg = data.get(DataType.PlanarSegmentation, scale=1)
    if visdom is not None and len(pred_seg) > 0 and len(gt_seg) > 0:
        pred_seg = torch.sigmoid(pred_seg[0][0]).cpu()
        gt_seg = gt_seg[0][0].cpu()
        visdom[0].heatmap(
            gt_seg.squeeze().flip(0),
            env=visdom[1],
            win='planar_seg_gt',
            opts=dict(
                title='Plannar vs Non-Plannar GT',
                caption='Plannar vs Non-Plannar GT',
                xmax=1,
                width=512,
                height=256))
        visdom[0].heatmap(
            pred_seg.squeeze().flip(0),
            env=visdom[1],
            win='planar_seg_pred',
            opts=dict(
                title='Plannar vs Non-Plannar Prediction',
                caption='Plannar vs Non-Plannar Prediction',
                xmax=1,
                width=512,
                height=256))

    pred_normals = output.get(DataType.Normals, scale=1)
    gt_normals = data.get(DataType.Normals, scale=1)
    if len(pred_normals) > 0 and len(gt_normals) > 0:
        pred_normals = pred_normals[0][0].cpu().detach()
        pred_normals = stackVerticaly(pred_normals).squeeze().flip(0).numpy()
        gt_normals = gt_normals[0][0].cpu().detach()
        gt_normals = stackVerticaly(gt_normals).squeeze().flip(0).numpy()
        normals_fig = heatmapGrid([gt_normals, pred_normals], ["Gt normals", "Pred normals"])
        if save_to_disk:
            py.io.write_image(normals_fig, osp.join(data_folder, 'normals.png'))
        if visdom is not None:
            visdom[0].plotlyplot(normals_fig, win="normals", env=visdom[1])

    pred_depth_normals = output.get(DataType.DepthNormals, scale=1)
    gt_depth2 = data.get(DataType.Depth, scale=1)
    if len(pred_depth_normals) > 0 and len(gt_depth2) > 0:
        gt_depth_normals = DepthToNormals(kernel_size=3, dilation=2, padding=2,
                                          height=256, width=512)(gt_depth2[0].cpu())
        pred_depth_normals = pred_depth_normals[0][0].cpu().detach()
        pred_depth_normals = stackVerticaly(pred_depth_normals).squeeze().flip(0).numpy()
        gt_depth_normals = gt_depth_normals[0].cpu().detach()
        gt_depth_normals = stackVerticaly(gt_depth_normals).squeeze().flip(0).numpy()
        normals_fig = heatmapGrid([gt_depth_normals, pred_depth_normals],
                                  ["Gt depth normals", "Pred depth normals"])
        if save_to_disk:
            py.io.write_image(normals_fig, osp.join(data_folder, 'depth_normals.png'))
        if visdom is not None:
            visdom[0].plotlyplot(normals_fig, win="depth_normals", env=visdom[1])

    pred_plane_param = output.get(DataType.PlaneParams, scale=1)
    gt_plane_param = data.get(DataType.PlaneParams, scale=1)
    if visdom is not None and len(pred_plane_param) > 0 and len(gt_normals) > 0:
        gt_plane_param = toPlaneParams(gt_normals[0], data.get(DataType.Depth, scale=1)[0])
        pred_plane_param = pred_plane_param[0][0].cpu()
        gt_plane_param = gt_plane_param[0].cpu()
        visdom[0].heatmap(
            stackVerticaly(gt_plane_param).squeeze().flip(0),
            env=visdom[1],
            win='plane_params_gt',
            opts=dict(
                title='Plane Params GT',
                caption='Plane Params GT',
                width=512,
                height=768))
        visdom[0].heatmap(
            stackVerticaly(pred_plane_param).squeeze().flip(0),
            env=visdom[1],
            win='plane_params_pred',
            opts=dict(
                title='Plane Params Prediction',
                caption='Plane Params Prediction',
                width=512,
                height=768))

    planes = data.get(DataType.Planes)
    if visdom is not None and len(planes) > 0:
        planes_data = planes[0][0].cpu()
        planes_data = visualizePlanes(planes_data).squeeze().flip(0)
        print(planes_data.size())
        visdom[0].image(
            planes_data,
            env=visdom[1],
            win='planes',
            opts=dict(
                title='Planes GT',
                caption='Planes GT'))

    rgb = data.get(DataType.Image, scale=1)[0][0, :, :, :].cpu().detach()
    rgb = rgb.permute(1, 2, 0).numpy() * 250
    pred_normals = output.get(DataType.Normals, scale=1)
    if len(pred_normals) > 0:
        pred_normals = pred_normals[0][0].cpu().detach()
        pred_normals = torch.squeeze(pred_normals.permute(1, 2, 0)).numpy()
    else:
        pred_normals = None

    for i, depth in enumerate(output.get(DataType.Depth, scale=1)):
        pred_depth = depth[0].cpu().detach()
        pred_depth = torch.squeeze(pred_depth.permute(1, 2, 0)).numpy()
        pcl = panoDepthToPcl(pred_depth, rgb, normals=pred_normals)
        savePcl(pcl[0], pcl[1], osp.join(data_folder, 'pcl_cloud_' + str(i) + '.ply'),
                normals=pcl[2])

    gt_depth = data.get(DataType.Depth, scale=1)[0][0].cpu().detach()
    gt_depth = torch.squeeze(gt_depth.permute(1, 2, 0)).numpy()

    pcl_gt = panoDepthToPcl(gt_depth, rgb)
    savePcl(pcl_gt[0], pcl_gt[1], osp.join(data_folder, 'pcl_gt_cloud.ply'))


@pytorchDetachedProcess
def visualize_loss(visdom, loss_trackers, directory, save_to_disk=False):
    '''
    Updates the loss visualization
    '''
    data_folder = osp.join(directory, "visdom")
    os.makedirs(data_folder, exist_ok=True)

    for key, tracker in loss_trackers.items():
        graph = linePlot([(key, tracker.x, tracker.y, None)])
        file = osp.join(data_folder, key + '.json')
        py.io.write_json(graph, file)

        if save_to_disk:
            py.io.write_image(graph, osp.join(data_folder, key + '.png'))
        if visdom is not None:
            visdom[0].plotlyplot(graph, win=key, env=visdom[1])
