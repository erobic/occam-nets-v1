from trainers.base_trainer import BaseTrainer
from trainers.occam_trainer import OccamTrainer
import torch
import torch.nn.functional as F


def spectral_decoupling_loss(batch, batch_idx, logits, group_by, lambdas, gammas):
    if group_by is not None:
        lambdas = [lambdas[int(gix)] for gix in batch[group_by]]
        gammas = [gammas[int(gix)] for gix in batch[group_by]]
    else:
        lambdas = [lambdas[0]] * len(batch['x'])
        gammas = [gammas[0]] * len(batch['x'])

    lambdas = torch.FloatTensor(lambdas).to(batch['x'].device)
    gammas = torch.FloatTensor(gammas).to(batch['x'].device)
    gt_logits = logits.gather(1, batch['y'].view(-1, 1)).squeeze()

    main_loss = F.cross_entropy(logits, batch['y'].squeeze(), reduction='none')

    l2_loss = (0.5 * lambdas * (gt_logits - gammas) ** 2)
    return main_loss, l2_loss


class SpectralDecouplingTrainer(BaseTrainer):
    """
    Implementation for:
    Pezeshki, Mohammad, et al. "Gradient Starvation: A Learning Proclivity in Neural Networks." arXiv preprint arXiv:2011.09468 (2020).
    The paper shows that decay and shift in network's logits can help decouple learning of features, which may enable learning of signal too.
    """

    def compute_main_loss(self, batch, batch_idx, model_out):
        main_loss, l2_loss = spectral_decoupling_loss(batch, batch_idx, model_out, self.trainer_cfg.group_by,
                                                      self.trainer_cfg.lambdas, self.trainer_cfg.gammas)
        self.log('main_loss', main_loss.mean(), on_epoch=True, py_logging=False)
        self.log('l2_loss', l2_loss.mean(), on_epoch=True, py_logging=False)
        return main_loss.mean() + l2_loss.mean()


class OccamSpectralDecouplingTrainer(OccamTrainer):
    """
    Implementation for:
    Pezeshki, Mohammad, et al. "Gradient Starvation: A Learning Proclivity in Neural Networks." arXiv preprint arXiv:2011.09468 (2020).
    The paper shows that decay and shift in network's logits can help decouple learning of features, which may enable learning of signal too.
    """

    def compute_main_loss(self, batch, batch_idx, model_out, exit_ix):
        main_loss, l2_loss = spectral_decoupling_loss(batch, batch_idx, model_out[f'E={exit_ix}, logits'],
                                                      self.trainer_cfg.group_by,
                                                      self.trainer_cfg.lambdas, self.trainer_cfg.gammas)
        self.log('main_loss', main_loss.mean(), on_epoch=True, py_logging=False)
        self.log('l2_loss', l2_loss.mean(), on_epoch=True, py_logging=False)
        return main_loss + l2_loss
