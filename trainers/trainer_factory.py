from trainers.base_trainer import *
from trainers.pgi_trainer import *
from trainers.group_upweighting_trainer import *
from trainers.spectral_decoupling_trainer import *
from trainers.occam_trainer import *


def build_trainer(cfg):
    return eval(cfg.trainer.name)(cfg)


def load_trainer(cfg):
    return eval(cfg.trainer.name).load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, config=cfg)
