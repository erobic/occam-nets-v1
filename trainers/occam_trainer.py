import logging
import os

from trainers.base_trainer import BaseTrainer
import torch
import torch.nn.functional as F
from utils.metrics import Accuracy
from models.occam_lib import MultiExitStats
from analysis.analyze_segmentation import SegmentationMetrics, save_exitwise_heatmaps
from utils.cam_utils import get_class_cams_for_occam_nets
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE

class OccamTrainer(BaseTrainer):
    """
    Implementation for: OccamNets
    """

    def __init__(self, config):
        super().__init__(config)
        # validation checks
        assert hasattr(self.model, 'multi_exit')
        self.num_exits = len(self.model.multi_exit.exit_block_nums)

    def training_step(self, batch, batch_idx):
        model_out = self(batch['x'])
        loss = 0

        # Compute exit-wise losses
        for exit_ix in range(len(self.model.multi_exit.exit_block_nums)):
            _loss_dict = self.compute_losses(batch, batch_idx, model_out, exit_ix)
            for _k in _loss_dict:
                self.log(f'{_k} E={exit_ix}', _loss_dict[_k].mean(), py_logging=False)
                loss += _loss_dict[_k].mean()
        return loss

    def compute_losses(self, batch, batch_idx, model_out, exit_ix):
        """
        Computes CAM Suppression loss, exit gate loss and gate-weighted CE Loss
        :param batch:
        :param batch_idx:
        :param model_out:
        :param exit_ix:
        :return:
        """
        gt_ys = batch['y'].squeeze()
        loss_dict = {}

        logits = model_out[f'E={exit_ix}, logits']

        ###############################################################################################################
        # Compute CAM suppression loss
        ###############################################################################################################
        supp_cfg = self.trainer_cfg.cam_suppression
        if supp_cfg.loss_wt != 0.0:
            loss_dict['supp'] = supp_cfg.loss_wt * CAMSuppressionLoss()(model_out[f'E={exit_ix}, cam'], gt_ys)

        ###############################################################################################################
        # Compute exit gate loss
        ###############################################################################################################
        gate_cfg = self.trainer_cfg.exit_gating
        if gate_cfg.loss_wt != 0.0:
            loss_name = f'ExitGateLoss_{exit_ix}'
            if batch_idx == 0:
                # The loss is stateful (maintains accuracy/gate_wt)
                if not hasattr(self, loss_name):
                    setattr(self, loss_name, ExitGateLoss(gate_cfg.train_acc_thresholds[exit_ix],
                                                          gate_cfg.balance_factor))
                getattr(self, loss_name).on_epoch_start()

            gates = model_out[f'E={exit_ix}, gates']
            force_use = (self.current_epoch + 1) <= gate_cfg.min_epochs
            loss_dict['gate'] = gate_cfg.loss_wt * getattr(self, loss_name) \
                (batch['item_ix'], logits, gt_ys, gates, force_use=force_use)

        ###############################################################################################################
        # Compute gate-weighted CE Loss
        ###############################################################################################################
        # self.compute_main_loss(batch_idx, batch, model_out, logits, gt_ys, exit_ix, loss_dict)
        gate_cfg = self.trainer_cfg.exit_gating
        loss_name = f"GateWeightedCELoss_{exit_ix}"
        if batch_idx == 0:
            setattr(self, loss_name, GateWeightedCELoss(gate_cfg.gamma0, gate_cfg.gamma, offset=gate_cfg.weight_offset))
        prev_gates = None if exit_ix == 0 else model_out[f"E={exit_ix - 1}, gates"]
        unweighted_loss = self.compute_main_loss(batch, batch_idx, model_out, exit_ix)
        assert len(unweighted_loss) == len(batch['y'])
        loss_dict['main'] = getattr(self, loss_name)(exit_ix, logits, prev_gates, gt_ys, unweighted_loss)
        return loss_dict

    def compute_main_loss(self, batch, batch_idx, model_out, exit_ix):
        return F.cross_entropy(model_out[f'E={exit_ix}, logits'], batch['y'].squeeze(), reduction='none')

    def shared_validation_step(self, batch, batch_idx, split, dataloader_idx=None, model_outputs=None):
        if model_outputs is None:
            model_outputs = self(batch['x'])
        super().shared_validation_step(batch, batch_idx, split, dataloader_idx, model_outputs)
        if batch_idx == 0:
            me_stats = MultiExitStats()
            setattr(self, f'{split}_{self.get_loader_key(split, dataloader_idx)}_multi_exit_stats', me_stats)
        me_stats = getattr(self, f'{split}_{self.get_loader_key(split, dataloader_idx)}_multi_exit_stats')

        me_stats(self.num_exits, model_outputs, batch['y'], batch['class_name'], batch['group_name'])

    def shared_validation_epoch_end(self, outputs, split):
        super().shared_validation_epoch_end(outputs, split)
        loader_keys = self.get_dataloader_keys(split)
        for loader_key in loader_keys:
            me_stats = getattr(self, f'{split}_{loader_key}_multi_exit_stats')
            self.log_dict(me_stats.summary(prefix=f'{split} {loader_key} '))

    def segmentation_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx=None):
        if 'mask' not in batch:
            return

        loader_key = self.get_loader_key(split, dataloader_idx)

        # Per-exit segmentation metrics
        for cls_type in ['gt', 'pred']:
            exit_to_class_cams = {}

            for exit_name in self.model.multi_exit.get_exit_names():
                metric_key = f'{cls_type}_{exit_name}_{split}_{loader_key}_segmentation_metrics'

                if batch_idx == 0:
                    setattr(self, metric_key, SegmentationMetrics())
                gt_masks = batch['mask']
                classes = batch['y'] if cls_type == 'gt' else model_out[f"{exit_name}, logits"].argmax(dim=-1)
                class_cams = get_class_cams_for_occam_nets(model_out[f"{exit_name}, cam"], classes)
                getattr(self, metric_key).update(gt_masks, class_cams)
                exit_to_class_cams[exit_name] = class_cams
            self.save_heat_maps_step(batch_idx, batch, exit_to_class_cams, heat_map_suffix=f"_{cls_type}")

    def save_heat_maps_step(self, batch_idx, batch, exit_to_heat_maps, heat_map_suffix=''):
        """
        Saves the original image, GT mask and the predicted CAMs for the first sample in the batch
        :param batch_idx:
        :param batch:
        :param exit_to_heat_maps:
        :return:
        """
        _exit_to_heat_maps = {}
        for en in exit_to_heat_maps:
            _exit_to_heat_maps[en] = exit_to_heat_maps[en][0]
        save_dir = os.path.join(os.getcwd(), f'viz/visualizations_ep{self.current_epoch}_b{batch_idx}')
        gt_mask = None if 'mask' not in batch else batch['mask'][0]
        save_exitwise_heatmaps(batch['x'][0], gt_mask, _exit_to_heat_maps, save_dir, heat_map_suffix=heat_map_suffix)

    def segmentation_metric_epoch_end(self, split, loader_key):
        for cls_type in ['gt', 'pred']:
            for exit_name in self.model.multi_exit.get_exit_names():
                metric_key = f'{cls_type}_{exit_name}_{split}_{loader_key}_segmentation_metrics'
                if hasattr(self, metric_key):
                    seg_metric_vals = getattr(self, metric_key).summary()
                    for sk in seg_metric_vals:
                        self.log(f"{cls_type} {split} {loader_key} {exit_name} {sk}", seg_metric_vals[sk])

    def on_save_checkpoint(self, checkpoint):
        for exit_ix in range(len(self.model.multi_exit.exit_block_nums)):
            getattr(self, f'ExitGateLoss_{exit_ix}').on_save_checkpoint(checkpoint, exit_ix)

    def on_load_checkpoint(self, checkpoint):
        for exit_ix in range(len(self.model.multi_exit.exit_block_nums)):
            try:
                getattr(self, f'ExitGateLoss_{exit_ix}').on_load_checkpoint(checkpoint, exit_ix)
            except:
                logging.getLogger().error(f"Could not load {f'ExitGateLoss_{exit_ix}'}")

    def accuracy_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx, accuracy):
        accuracy.update(model_out['E=early, logits'], batch['y'], batch['class_name'], batch['group_name'])

    def init_calibration_analysis(self, split, loader_key):
        setattr(self, f'{split}_{loader_key}_calibration_analysis', CalibrationAnalysis(self.num_exits))


class GateWeightedCELoss():
    def __init__(self, gamma0=3, gamma=1, eps=1e-5, offset=0.1):
        self.gamma0 = gamma0
        self.gamma = gamma
        self.eps = eps
        self.offset = offset
        self.max_wt = 0  # stateful

    def __call__(self, exit_ix, curr_logits, prev_gates, gt_ys, unweighted_loss):
        curr_gt_proba = F.softmax(curr_logits, dim=1).gather(1, gt_ys.squeeze().view(-1, 1)).squeeze()
        if exit_ix == 0:
            assert prev_gates is None
            # bias-amp loss
            loss_wt = curr_gt_proba.detach() ** self.gamma0
        else:
            # weighted loss
            loss_wt = (1 - prev_gates.detach()) ** self.gamma
        curr_max_wt = loss_wt.max().detach()
        if curr_max_wt > self.max_wt:
            self.max_wt = curr_max_wt

        loss_wt = loss_wt / (self.max_wt + self.eps)
        return (loss_wt + self.offset) * unweighted_loss


class CAMSuppressionLoss():
    """
    KLD loss between uniform distribution and inconfident CAM cell locations (inconfident towards GT class)
    Inconfident regions are hard thresholded with mean CAM value
    """

    def __call__(self, cams, gt_ys):
        b, c, h, w = cams.shape
        cams = cams.reshape((b, c, h * w))
        gt_cams = torch.gather(cams, dim=1, index=gt_ys.squeeze().unsqueeze(dim=1).unsqueeze(dim=2)
                               .repeat(1, 1, h * w)).squeeze().reshape((b, h * w))
        gt_max, gt_min, gt_mean = torch.max(gt_cams, dim=1)[0], torch.min(gt_cams, dim=1)[0], torch.mean(gt_cams, dim=1)
        norm_gt_cams = (gt_cams - gt_min.unsqueeze(1)) / (gt_max.unsqueeze(1) - gt_min.unsqueeze(1)).detach()
        threshold = gt_mean.unsqueeze(1).repeat(1, h * w)

        # Assign weights so that the locations which have a score lower than the threshold are suppressed
        supp_wt = torch.where(gt_cams > threshold, torch.zeros_like(norm_gt_cams), torch.ones_like(norm_gt_cams))

        uniform_targets = torch.ones_like(cams) / c
        uniform_kld_loss = torch.sum(uniform_targets * (torch.log_softmax(uniform_targets, dim=1) -
                                                        torch.log_softmax(cams, dim=1)), dim=1)
        supp_loss = (supp_wt * uniform_kld_loss).mean()
        return supp_loss


class ExitGateLoss():
    """
    Trains the gate to exit if the sample was correctly predicted and if the overall accuracy is lower than the threshold
    """

    def __init__(self, acc_threshold, balance_factor=0.5):
        self.acc_threshold = acc_threshold
        self.accuracy = Accuracy()
        self.balance_factor = balance_factor
        self.item_to_correctness = {}

    def __call__(self, item_ixs, logits, gt_ys, gates, force_use=False, eps=1e-5):
        """

        :param logits:
        :param gt_ys:
        :param gates: probability of exiting predicted by the gate
        :param force_use:
        :param eps:
        :return:
        """
        pred_ys = torch.argmax(logits, dim=1)
        self.accuracy.update(logits, gt_ys, gt_ys, gt_ys)
        mpg = self.accuracy.get_mean_per_group_accuracy('class', topK=1)

        if mpg <= self.acc_threshold or force_use:
            gate_gt = (pred_ys == gt_ys.squeeze()).long().type(gates.type())
            for item_ix, is_correct in zip(item_ixs, gate_gt):
                self.item_to_correctness[int(item_ix)] = float(is_correct)
        else:
            gate_gt = torch.FloatTensor(
                [self.item_to_correctness[int(item_ix)] for item_ix in item_ixs]).type(gates.type())
        _exit_cnt, _continue_cnt = gate_gt.sum().detach(), (1 - gate_gt).sum().detach()
        # Assign balanced weights to exit vs continue preds
        _max_cnt = max(_exit_cnt, _continue_cnt)
        _exit_cnt, _continue_cnt = _exit_cnt / _max_cnt, _continue_cnt / _max_cnt
        _gate_loss_wts = torch.where(gate_gt > 0,
                                     (torch.ones_like(gate_gt) / (_exit_cnt + eps)) ** self.balance_factor,
                                     (torch.ones_like(gate_gt) / (_continue_cnt + eps)) ** self.balance_factor)

        gate_loss = _gate_loss_wts * F.binary_cross_entropy(gates, gate_gt, reduction='none')
        # gate_loss = _gate_loss_wts * F.binary_cross_entropy_with_logits(inv_sigmoid(gates), gate_gt, reduction='none')
        # gate_loss = _gate_loss_wts * F.mse_loss(gates, gate_gt, reduction='none')
        return gate_loss.mean()

    def on_epoch_start(self):
        self.accuracy = Accuracy()

    def on_save_checkpoint(self, checkpoint, exit_ix):
        checkpoint[f'item_to_correctness_{exit_ix}'] = self.item_to_correctness

    def on_load_checkpoint(self, checkpoint, exit_ix):
        self.item_to_correctness = checkpoint[f'item_to_correctness_{exit_ix}']


class CalibrationAnalysis():
    def __init__(self, num_exits):
        self.num_exits = num_exits
        self.exit_ix_to_logits, self.gt_ys = {}, None

    def update(self, batch, exit_outs):
        """
        Gather per-exit + overall logits
        """
        overall_logits = 0
        for exit_ix in range(self.num_exits):
            cam = exit_outs[f'E={exit_ix}, cam']
            logits = F.adaptive_avg_pool2d(cam, (1)).squeeze().detach().cpu()
            overall_logits += logits

            if f'E={exit_ix}' not in self.exit_ix_to_logits:
                self.exit_ix_to_logits[f'E={exit_ix}'] = logits
                self.exit_ix_to_logits[f'sum_upto_E={exit_ix}'] = overall_logits
            else:
                self.exit_ix_to_logits[f'E={exit_ix}'] = torch.cat([self.exit_ix_to_logits[f'E={exit_ix}'], logits],
                                                                   dim=0)
                self.exit_ix_to_logits[f'sum_upto_E={exit_ix}'] = torch.cat(
                    [self.exit_ix_to_logits[f'sum_upto_E={exit_ix}'], overall_logits], dim=0)

        if self.gt_ys is None:
            self.gt_ys = batch['y'].detach().cpu().squeeze()
        else:
            self.gt_ys = torch.cat([self.gt_ys, batch['y'].detach().cpu().squeeze()], dim=0)

    def plot_reliability_diagram(self, save_dir, bins=10):
        diagram = ReliabilityDiagram(bins)
        gt_ys = self.gt_ys.numpy()
        os.makedirs(save_dir, exist_ok=True)

        for exit_ix in self.exit_ix_to_logits:
            curr_conf = torch.softmax(self.exit_ix_to_logits[exit_ix].float(), dim=1).numpy()
            ece = ECE(bins).measure(curr_conf, gt_ys)
            diagram.plot(curr_conf, gt_ys, filename=os.path.join(save_dir, f'{exit_ix}.png'),
                         title_suffix=f' ECE={ece}')
