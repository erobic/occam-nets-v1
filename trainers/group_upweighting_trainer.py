from trainers.base_trainer import BaseTrainer
from trainers.occam_trainer import OccamTrainer
import torch
import numpy as np
import logging
import torch.nn.functional as F


def get_group_weights(loader, group_by, gamma):
    logging.getLogger().info("Initializing the group weights...")
    group_ix_to_cnt = {}
    group_ix_to_weight = {}
    total_samples = 0
    for batch in loader:
        for grp_ix in batch[group_by]:
            grp_ix = int(grp_ix)
            if grp_ix not in group_ix_to_cnt:
                group_ix_to_cnt[grp_ix] = 0
            group_ix_to_cnt[grp_ix] += 1
            total_samples += 1
    for group_ix in group_ix_to_cnt:
        group_ix_to_weight[group_ix] = (1 / group_ix_to_cnt[group_ix]) ** gamma
    return group_ix_to_weight


def compute_group_upweighting_loss(batch, batch_idx, logits, group_by, group_ix_to_weight):
    unweighted_loss = F.cross_entropy(logits, batch['y'].squeeze(), reduction='none')
    weights = torch.FloatTensor([group_ix_to_weight[int(group_ix)] for group_ix in batch[group_by]]).to(
        batch['x'].device)
    loss = weights * unweighted_loss
    return loss


class GroupUpweightingTrainer(BaseTrainer):
    """
    Simple upweighting technique which multiplies the loss by inverse group frequency.
    This has been found to work well when models are sufficiently underparameterized (e.g., low learning rate, high weight decay, fewer model parameters etc)
    Paper that investigated underparameterization with upweighting method: https://arxiv.org/abs/2005.04345
    """

    def compute_main_loss(self, batch, batch_idx, model_out):
        if not hasattr(self, 'group_ix_to_weight'):
            self.group_ix_to_weight = get_group_weights(self.train_loader,
                                                        group_by=self.trainer_cfg.group_by,
                                                        gamma=self.trainer_cfg.gamma)
        loss = compute_group_upweighting_loss(batch, batch_idx, model_out, self.trainer_cfg.group_by,
                                              self.group_ix_to_weight)
        return loss.mean()


class OccamGroupUpweightingTrainer(OccamTrainer):
    """
    Simple upweighting technique which multiplies the loss by inverse group frequency.
    This has been found to work well when models are sufficiently underparameterized (e.g., low learning rate, high weight decay, fewer model parameters etc)
    Paper that investigated underparameterization with upweighting method: https://arxiv.org/abs/2005.04345
    """

    def compute_main_loss(self, batch, batch_idx, model_out, exit_ix):
        if not hasattr(self, 'group_ix_to_weight'):
            self.group_ix_to_weight = get_group_weights(self.train_loader,
                                                        group_by=self.trainer_cfg.group_by,
                                                        gamma=self.trainer_cfg.gamma)
        loss = compute_group_upweighting_loss(batch, batch_idx, model_out[f'E={exit_ix}, logits'],
                                              self.trainer_cfg.group_by,
                                              self.group_ix_to_weight)
        return loss
