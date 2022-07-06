import torch
from typing import Any, cast, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import ReduceLROnPlateau


# https://raw.githubusercontent.com/ildoonet/pytorch-gradual-warmup-lr/master/warmup_scheduler/scheduler.py
class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, start_lr, total_epoch, after_scheduler=None):
        self.start_lr = start_lr
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(LinearWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            return self.after_scheduler.get_last_lr()

        def _get_lr(lr):
            lr = (lr - self.start_lr) / self.total_epoch * (self.last_epoch + 1)
            return lr

        return [_get_lr(base_lr) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.last_epoch >= self.total_epoch and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(LinearWarmupScheduler, self).step(epoch)


def build_lr_scheduler(optim_cfg, optim):
    lr_cfg = optim_cfg.lr_scheduler
    lr_args = lr_cfg.args
    lr_scheduler = eval(lr_cfg.type)(optim, **lr_args)
    if optim_cfg.lr_warmup.epochs > 0:
        lr_scheduler = LinearWarmupScheduler(optim,
                                             start_lr=optim_cfg.lr_warmup.start_lr,
                                             total_epoch=optim_cfg.lr_warmup.epochs,
                                             after_scheduler=lr_scheduler)
    return lr_scheduler
