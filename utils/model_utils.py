import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)['state_dict']
    try:
        model.load_state_dict(state_dict)
        return model
    except:
        model = ModelWrapper(model)
        model.load_state_dict(state_dict)
        return model.model
