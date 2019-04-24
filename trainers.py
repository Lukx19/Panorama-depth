import torch
import torch.nn.functional as F
import time

import math
import shutil
import os.path as osp

import util
from metrics import (abs_rel_error, delta_inlier_ratio,
                     lin_rms_sq_error, log_rms_sq_error, sq_rel_error)

from util import saveTensorDepth, toDevice
import torchvision.transforms as Tt
from torchvision.utils import make_grid

from annotated_data import AnnotatedData, DataType
import colorsys
import numpy as np

# From https://github.com/fyu/drn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {'val': self.val,
                'sum': self.sum,
                'count': self.count,
                'avg': self.avg}

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


def resetAverageMeters(average_meters):
    for key, val in average_meters.items():
        val.reset()


def loadAverageMeters(average_meters):
    loaded_meters = {}
    for key, val in average_meters.items():
        loaded_meters[key] = AverageMeter()
        loaded_meters.from_dict(val)
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


def visualizeChannels(tensor):
    '''Stack channels vertically'''
    ch, h, w = tensor.size()
    tensor = tensor.reshape((1, -1, w))
    return tensor


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
    gt_depth_1x = data.get(DataType.Depth, scale=1)[0]

    gt_depth_2x = F.interpolate(gt_depth_1x, scale_factor=0.5)
    gt_depth_4x = F.interpolate(gt_depth_1x, scale_factor=0.25)
    mask_1x = data.get(DataType.Mask, scale=1)[0]
    mask_2x = F.interpolate(mask_1x, scale_factor=0.5)
    mask_4x = F.interpolate(mask_1x, scale_factor=0.25)

    inputs = {DataType.Image: rgb}

    data.add(rgb, DataType.Image, scale=1)
    data.add(gt_depth_1x, DataType.Depth, scale=1)
    data.add(gt_depth_2x, DataType.Depth, scale=2)
    data.add(gt_depth_4x, DataType.Depth, scale=4)
    data.add(mask_1x, DataType.Mask, scale=1)
    data.add(mask_2x, DataType.Mask, scale=2)
    data.add(mask_4x, DataType.Mask, scale=4)
    return inputs, data


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

    inputs[DataType.SparseDepth] = sparse_scales
    data.add(sparse_depth_1x, DataType.SparseDepth, scale=1)
    data.add(sparse_depth_2x, DataType.SparseDepth, scale=2)
    data.add(sparse_depth_4x, DataType.SparseDepth, scale=4)

    return inputs, data


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
        gt = d.get(DataType.Depth, scale=1)[0][i]
        prediction = o[i]
        img = d.get(DataType.Image, scale=1)[0][i]

        saveTensorDepth(name + "_gt_depth", gt, 1)
        saveTensorDepth(name + "_pred_depth", prediction, 1)

        sparse_points = d.get(DataType.SparseDepth, scale=1)
        if len(sparse_points) > 0:
            sparse_points = sparse_points[0][i]
            saveTensorDepth(name + "_sparse_depth",
                            sparse_points, unnorm_depth)

        color_img = Tt.functional.to_pil_image(img.cpu())
        color_img.save(name + "_color.jpg")

        # d['original_image'][i].write(osp.join(results_dir, name + "_color.jpg"))


def totalLoss(loss):
    total_loss = 0
    if isinstance(loss, dict):
        for key, val in loss.items():
            total_loss += val
    elif isinstance(loss, list):
        for val in loss.items():
            total_loss += val
    elif isinstance(loss, torch.Tensor):
        total_loss = loss
    else:
        raise ValueError("Unsupoted loss type: ", type(loss))
    return total_loss


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
            sum_losses_fce=totalLoss,
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
        self.abs_rel_error_meter = AverageMeter()
        self.sq_rel_error_meter = AverageMeter()
        self.lin_rms_sq_error_meter = AverageMeter()
        self.log_rms_sq_error_meter = AverageMeter()
        self.d1_inlier_meter = AverageMeter()
        self.d2_inlier_meter = AverageMeter()
        self.d3_inlier_meter = AverageMeter()

        # Track the best inlier ratio recorded so far
        self.best_d1_inlier = 0.0
        self.is_best = False

        # List of length 2 [Visdom instance, env]
        self.vis = visdom

        # Loss trackers
        self.loss_meters = {
            "total": AverageMeter()
        }
        self.dry_run = False

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
        self.loss_meters["total"].update(total_loss)
        if isinstance(raw_loss, dict):
            for key, val in raw_loss.items():
                if key not in self.loss_meters:
                    self.loss_meters[key] = AverageMeter()
                self.loss_meters[key].update(val)

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
            raw_loss = self.compute_loss(output, data)
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
                self.save_checkpoint()
            # Every few batches
            if batch_num % self.visualization_freq == 0:

                # Visualize the loss
                self.visualize_loss(batch_num, self.loss_meters)
                self.visualize_samples(data, output)

                # Print the most recent batch report
                self.print_batch_report(batch_num)
                resetAverageMeters(self.loss_meters)

    def validate(self, checkpoint_path=None, save_all_predictions=False):
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
            for batch_num, raw_data in enumerate(self.val_dataloader):

                # Parse the data
                inputs, data = self.parse_data(raw_data)
                data = data.to(self.device)
                inputs = toDevice(inputs, self.device)

                # Run a forward pass
                output = self.annotate_outputs(self.forward_pass(inputs))

                # Compute the evaluation metrics
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
        return report

    def train(self, checkpoint_path=None, weights_only=False):
        print('Starting training')

        # Load pretrained parameters if desired
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, weights_only)
            if weights_only:
                self.initialize_visualizations()
        else:
            # Initialize any training visualizations
            self.initialize_visualizations()

        # Train for specified number of epochs
        for self.epoch in range(self.epoch, self.num_epochs):

            # Increment the LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Run an epoch of training
            self.train_one_epoch()

            if self.epoch % self.validation_freq == 0:
                self.validate()
                self.save_checkpoint()
                self.visualize_metrics()

    def reset_eval_metrics(self):
        '''
        Resets metrics used to evaluate the model
        '''
        self.abs_rel_error_meter.reset()
        self.sq_rel_error_meter.reset()
        self.lin_rms_sq_error_meter.reset()
        self.log_rms_sq_error_meter.reset()
        self.d1_inlier_meter.reset()
        self.d2_inlier_meter.reset()
        self.d3_inlier_meter.reset()
        self.is_best = False

    def compute_eval_metrics(self, output, data):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output.get(DataType.Depth, scale=1)[0]

        gt_depth = data.get(DataType.Depth, scale=1)[0]
        depth_mask = data.get(DataType.Depth, scale=1)[0]
        sparse_pts = data.get(DataType.SparseDepth, scale=1)[0]
        if len(sparse_pts) > 0:
            pts_present_mask = torch.sign(sparse_pts[0])
            depth_mask = depth_mask - depth_mask * pts_present_mask

        N = depth_mask.sum()

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

        self.abs_rel_error_meter.update(abs_rel, N)
        self.sq_rel_error_meter.update(sq_rel, N)
        self.lin_rms_sq_error_meter.update(rms_sq_lin, N)
        self.log_rms_sq_error_meter.update(rms_sq_log, N)
        self.d1_inlier_meter.update(d1, N)
        self.d2_inlier_meter.update(d2, N)
        self.d3_inlier_meter.update(d3, N)

    def load_checkpoint(self, checkpoint_path=None, weights_only=False, eval_mode=False):
        '''
        Initializes network with pretrained parameters
        '''
        if checkpoint_path is not None:
            print('Loading checkpoint \'{}\''.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

            # If we want to continue training where we left off, load entire training state
            if not weights_only:
                self.epoch = checkpoint['epoch']
                experiment_name = checkpoint['experiment']
                self.vis[1] = experiment_name
                self.best_d1_inlier = checkpoint['best_d1_inlier']
                if "loss_meter" in checkpoint:
                    self.loss_meters["total"].from_dict(checkpoint["loss_meter"])
                if "loss_meters" in checkpoint:
                    self.loss_meters = loadAverageMeters(checkpoint["loss_meters"])

            else:
                print('NOTE: Loading weights only')

            # Load the optimizer and model state
            if not eval_mode:
                util.load_optimizer(
                    self.optimizer,
                    checkpoint['optimizer'],
                    self.device)
            util.load_partial_model(
                self.network,
                checkpoint['state_dict'])

            print('Loaded checkpoint \'{}\' (epoch {})'.format(
                checkpoint_path, checkpoint['epoch']))
        else:
            print('WARNING: No checkpoint found')

    def initialize_visualizations(self):
        '''
        Initializes visualizations
        '''
        # if self.vis[1] == self.vis[0].get_env_list():
        self.vis[0].delete_env(self.vis[1])
        self.vis[0].line(
            env=self.vis[1],
            X=torch.zeros(1, 4).long(),
            Y=torch.zeros(1, 4).float(),
            win='error_metrics',
            opts=dict(
                title='Depth Error Metrics',
                markers=True,
                xlabel='Epoch',
                ylabel='Error',
                legend=['Abs. Rel. Error', 'Sq. Rel. Error', 'Linear RMS Error', 'Log RMS Error']))

        self.vis[0].line(
            env=self.vis[1],
            X=torch.zeros(1, 3).long(),
            Y=torch.zeros(1, 3).float(),
            win='inlier_metrics',
            opts=dict(
                title='Depth Inlier Metrics',
                markers=True,
                xlabel='Epoch',
                ylabel='Percent',
                legend=['d1', 'd2', 'd3']))

    def visualize_loss(self, batch_num, loss_meters):
        '''
        Updates the loss visualization
        '''
        total_num_batches = self.epoch * len(self.train_dataloader) + batch_num
        for key, loss_meter in loss_meters.items():
            if not self.vis[0].win_exists(key, env=self.vis[1]):
                self.vis[0].line(
                    env=self.vis[1],
                    X=torch.zeros(1, 1).long(),
                    Y=torch.zeros(1, 1).float(),
                    win=key,
                    opts=dict(
                        title=key + ' Plot',
                        markers=False,
                        xlabel='Iteration',
                        ylabel='Loss',
                        legend=[key]))
            else:
                self.vis[0].line(
                    env=self.vis[1],
                    X=torch.tensor([total_num_batches]),
                    Y=torch.tensor([loss_meter.avg]),
                    win=key,
                    update='append',
                    opts=dict(
                        legend=[key]))

    def visualize_samples(self, data, output):
        '''
        Updates the output samples visualization
        '''
        rgb = data.get(DataType.Image, scale=1)[0][0, :, :, :].cpu()
        depth_pred = output.get(DataType.Depth, scale=1)[0][0].cpu()

        gt_depth = data.get(DataType.Depth, scale=1)[0][0].cpu()
        depth_mask = data.get(DataType.Mask, scale=1)[0][0].cpu()

        self.vis[0].image(
            visualize_rgb(rgb),
            env=self.vis[1],
            win='rgb',
            opts=dict(
                title='Input RGB Image',
                caption='Input RGB Image'))

        self.vis[0].image(
            visualize_mask(depth_mask),
            env=self.vis[1],
            win='mask',
            opts=dict(
                title='Mask',
                caption='Mask'))

        max_depth = max(
            ((depth_mask > 0).float() * gt_depth).max().item(),
            ((depth_mask > 0).float() * depth_pred).max().item())
        self.vis[0].heatmap(
            depth_pred.squeeze().flip(0),
            env=self.vis[1],
            win='depth_pred',
            opts=dict(
                title='Depth Prediction',
                caption='Depth Prediction',
                xmax=max_depth,
                xmin=gt_depth.min().item(),
                width=512,
                height=256))

        self.vis[0].heatmap(
            gt_depth.squeeze().flip(0),
            env=self.vis[1],
            win='gt_depth',
            opts=dict(
                title='Depth GT',
                caption='Depth GT',
                xmax=max_depth,
                width=512,
                height=256))
        pred_seg = output.get(DataType.PlanarSegmentation, scale=1)
        gt_seg = data.get(DataType.PlanarSegmentation, scale=1)
        if len(pred_seg) > 0 and len(gt_seg) > 0:
            pred_seg = torch.sigmoid(pred_seg[0][0]).cpu()
            gt_seg = gt_seg[0][0].cpu()
            self.vis[0].heatmap(
                gt_seg.squeeze().flip(0),
                env=self.vis[1],
                win='planar_seg_gt',
                opts=dict(
                    title='Plannar vs Non-Plannar GT',
                    caption='Plannar vs Non-Plannar GT',
                    xmax=1,
                    width=512,
                    height=256))
            self.vis[0].heatmap(
                pred_seg.squeeze().flip(0),
                env=self.vis[1],
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
            pred_normals = pred_normals[0][0].cpu()
            gt_normals = gt_normals[0][0].cpu()
            self.vis[0].heatmap(
                visualizeChannels(gt_normals).squeeze(),
                env=self.vis[1],
                win='normals_gt',
                opts=dict(
                    title='Normals GT',
                    caption='Normals GT',
                    width=512,
                    height=768))
            self.vis[0].heatmap(
                visualizeChannels(pred_normals).squeeze(),
                env=self.vis[1],
                win='normals_pred',
                opts=dict(
                    title='Normals Prediction',
                    caption='Normals Prediction',
                    width=512,
                    height=768))
        planes = data.get(DataType.Planes)
        if len(planes) > 0:
            planes_data = planes[0][0].cpu()
            planes_data = visualizePlanes(planes_data).squeeze().flip(0)
            print(planes_data.size())
            self.vis[0].image(
                planes_data,
                env=self.vis[1],
                win='planes',
                opts=dict(
                    title='Planes GT',
                    caption='Planes GT'))

    def visualize_metrics(self):
        '''
        Updates the metrics visualization
        '''
        abs_rel = self.abs_rel_error_meter.avg
        sq_rel = self.sq_rel_error_meter.avg
        lin_rms = math.sqrt(self.lin_rms_sq_error_meter.avg)
        log_rms = math.sqrt(self.log_rms_sq_error_meter.avg)
        d1 = self.d1_inlier_meter.avg
        d2 = self.d2_inlier_meter.avg
        d3 = self.d3_inlier_meter.avg

        errors = torch.FloatTensor([abs_rel, sq_rel, lin_rms, log_rms])
        errors = errors.view(1, -1)
        epoch_expanded = torch.ones(errors.shape) * (self.epoch + 1)
        self.vis[0].line(
            env=self.vis[1],
            X=epoch_expanded,
            Y=errors,
            win='error_metrics',
            update='append',
            opts=dict(
                legend=['Abs. Rel. Error', 'Sq. Rel. Error', 'Linear RMS Error', 'Log RMS Error']))

        inliers = torch.FloatTensor([d1, d2, d3])
        inliers = inliers.view(1, -1)
        epoch_expanded = torch.ones(inliers.shape) * (self.epoch + 1)
        self.vis[0].line(
            env=self.vis[1],
            X=epoch_expanded,
            Y=inliers,
            win='inlier_metrics',
            update='append',
            opts=dict(
                legend=['d1', 'd2', 'd3']))

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
        report = ('Epoch: {}\n'
                  '  Avg. Abs. Rel. Error: {:.4f}\n'
                  '  Avg. Sq. Rel. Error: {:.4f}\n'
                  '  Avg. Lin. RMS Error: {:.4f}\n'
                  '  Avg. Log RMS Error: {:.4f}\n'
                  '  Inlier D1: {:.4f}\n'
                  '  Inlier D2: {:.4f}\n'
                  '  Inlier D3: {:.4f}\n\n'.format(
                      self.epoch + 1,
                      self.abs_rel_error_meter.avg,
                      self.sq_rel_error_meter.avg,
                      math.sqrt(self.lin_rms_sq_error_meter.avg),
                      math.sqrt(self.log_rms_sq_error_meter.avg),
                      self.d1_inlier_meter.avg,
                      self.d2_inlier_meter.avg,
                      self.d3_inlier_meter.avg))

        print(report)
        # Also update the best state tracker
        if self.best_d1_inlier < self.d1_inlier_meter.avg:
            self.best_d1_inlier = self.d1_inlier_meter.avg
            self.is_best = True
        return report

    def save_checkpoint(self):
        '''
        Saves the model state
        '''
        # Save latest checkpoint (constantly overwriting itself)
        checkpoint_path = osp.join(
            self.checkpoint_dir,
            'checkpoint_latest.pth')

        # Actually saves the latest checkpoint and also updating the file holding the best one
        util.save_checkpoint(
            {
                'epoch': self.epoch + 1,
                'experiment': self.name,
                'state_dict': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss_meters': serializeAverageMeters(self.loss_meters),
                'best_d1_inlier': self.best_d1_inlier
            },
            self.is_best,
            filename=checkpoint_path)

        # Copies the latest checkpoint to another file stored for each epoch
        history_path = osp.join(
            self.checkpoint_dir,
            'checkpoint_{:03d}.pth'.format(self.epoch + 1))
        shutil.copyfile(checkpoint_path, history_path)
        print('Checkpoint saved')
