import logging
import os
import torch
# from base_runner import *
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
from datasets import dataloader_factory
from trainers import trainer_factory
import pytorch_lightning as pl

log = logging.getLogger(__name__)


def init_seeds(cfg):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)


def init_expt_dir(cfg, expt_dir):
    cfg.expt_dir = expt_dir
    os.makedirs(cfg.expt_dir, exist_ok=True)


def init_app(cfg):
    # Dataloaders get stuck when num_workers > 0, this fixed it for me:
    # See: https://github.com/pytorch/pytorch/issues/1355#issuecomment-819203114
    torch.multiprocessing.set_sharing_strategy('file_system')

    init_seeds(cfg)

    logging.getLogger().info(f"Expt Dir: {os.getcwd()}")
    cfg.expt_dir = os.getcwd()


@hydra.main(config_path="conf", config_name="main_config")
def exec(cfg: DictConfig) -> None:
    init_app(cfg)
    cfg.model.num_classes = cfg.dataset.num_classes
    log.info(OmegaConf.to_yaml(cfg, sort_keys=True, resolve=True))

    # Commented out unsupported/yet-to-support tasks
    if cfg.task.name == 'test':
        loader = dataloader_factory.build_dataloader_for_split(cfg, cfg.data_sub_split)
        trainer = trainer_factory.load_trainer(cfg)
        trainer.set_dataloader_keys(cfg.data_split, [cfg.data_sub_split])
        pl_trainer = pl.Trainer(gpus=cfg.gpus,
                                limit_train_batches=cfg.trainer.limit_train_batches,
                                limit_val_batches=cfg.trainer.limit_val_batches,
                                limit_test_batches=cfg.trainer.limit_test_batches)
        pl_trainer.test(trainer, [loader])
    else:
        data_loaders = dataloader_factory.build_dataloaders(cfg)
        trainer = trainer_factory.build_trainer(cfg)
        trainer.set_dataloader_keys('val', list(data_loaders['val'].keys()))
        trainer.set_dataloader_keys('test', list(data_loaders['test'].keys()))
        trainer.set_iters_per_epoch(len(data_loaders['train']))

        pl_trainer = pl.Trainer(gpus=cfg.gpus,
                                min_epochs=cfg.optimizer.epochs,
                                max_epochs=cfg.optimizer.epochs,
                                check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
                                num_sanity_val_steps=0,
                                limit_train_batches=cfg.trainer.limit_train_batches,
                                limit_val_batches=cfg.trainer.limit_val_batches,
                                limit_test_batches=cfg.trainer.limit_test_batches,
                                precision=cfg.trainer.precision,
                                gradient_clip_val=cfg.trainer.gradient_clip_val,
                                log_every_n_steps=1)
        trainer.set_train_loader(data_loaders['train'])
        pl_trainer.fit(trainer,
                       train_dataloaders=data_loaders['train'],
                       val_dataloaders=list(data_loaders['val'].values()))
        pl_trainer.test(trainer, list(data_loaders['test'].values()))


ROOT = '/hdd/robik'

if __name__ == "__main__":
    exec()
