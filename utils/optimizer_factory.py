import torch.optim as optim
import logging
from torch.optim import *


def build_optimizer(optimizer_name, optim_args, named_params, freeze_layers=None,
                    custom_lr_config=None, model=None):  # , use_agc=False, agc_ignore_layers=['fc'], model=None):
    def should_be_added(layer_name, param):
        if not param.requires_grad:
            logging.getLogger().info(f'layer_name {layer_name} does not require grad')
            return False
        ret = True
        if freeze_layers is None:
            return ret
        for freeze_layer in freeze_layers:
            if layer_name.startswith(freeze_layer):
                ret = False
        return ret

    filt_params = []
    for name, param in named_params:
        param_dict = None
        if should_be_added(name, param):
            if custom_lr_config is not None:
                for custom_lr_name in custom_lr_config:
                    if name.startswith(custom_lr_name):
                        param_dict = {'params': param, 'lr': custom_lr_config[custom_lr_name]}
            if param_dict is None:
                param_dict = {'params': param, 'lr': optim_args.lr}
            filt_params.append(param_dict)
            logging.getLogger().debug(f"Added to optimizer: {name}")
        else:
            param.requires_grad = False  # for efficiency
            logging.getLogger().info(f"Removed from optimizer: {name}")

    optim = eval(optimizer_name)(filt_params, **optim_args)
    # if use_agc:
    #     if agc_ignore_layers is None or len(agc_ignore_layers) == 0:
    #         optim = AGC(model.parameters(), optim)  # TODO: take care of filtered params
    #     else:
    #         optim = AGC(filt_params, optim, model=model, ignore_agc=agc_ignore_layers)
    #     for group in optim.agc_params:
    #         for p in group['params']:
    #             if isinstance(p, dict):
    #                 print(p['params'])
    return optim
