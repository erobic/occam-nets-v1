import collections
import torch
from torch.autograd import Variable
from torch.utils.data import Sampler
from json import JSONEncoder
import numpy as np


# https://raw.githubusercontent.com/Cadene/bootstrap.pytorch/master/bootstrap/datasets/transforms.py

class Compose(object):
    """Composes several collate together.

    Args:
        transforms (list of ``Collate`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch):
        for transform in self.transforms:
            batch = transform(batch)
        return batch


class ListDictsToDictLists(object):

    def __init__(self):
        pass

    def __call__(self, batch):
        batch = self.ld_to_dl(batch)
        return batch

    def ld_to_dl(self, batch):
        if isinstance(batch[0], collections.Mapping):
            return {key: self.ld_to_dl([d[key] for d in batch]) for key in batch[0]}
        else:
            return batch


class StackTensors(object):

    def __init__(self, use_shared_memory=False, avoid_keys=[]):
        self.use_shared_memory = use_shared_memory
        self.avoid_keys = avoid_keys

    def __call__(self, batch):
        batch = self.stack_tensors(batch)
        return batch

    # key argument is useful for debuging
    def stack_tensors(self, batch, key=None):
        if isinstance(batch, collections.Mapping):
            out = {}
            for key, value in batch.items():
                if key not in self.avoid_keys:
                    out[key] = self.stack_tensors(value, key=key)
                else:
                    out[key] = value
            return out
        elif isinstance(batch, collections.Sequence) and torch.is_tensor(batch[0]):
            out = None
            if self.use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
        else:
            return batch


def dict_collate_fn():
    return Compose([
        ListDictsToDictLists(),
        StackTensors()
    ])


class NpEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def to_numpy_img(tensor):
    if len(tensor.shape) == 3:
        if tensor.shape[-1] != 3 and tensor.shape[0] == 3:
            tensor = torch.permute(tensor, (1, 2, 0))

    return (tensor.detach().cpu()).numpy()


def get_dir(file_path):
    return '/'.join(file_path.split('/')[:-1])


def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)
