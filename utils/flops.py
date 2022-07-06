from torchsummary import summary
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from models.occam_resnet import *
from models.variable_width_resnet import *
from models.occam_resnet import occam_resnet18


class PCAConfig():
    def __init__(self):
        self.file = None


class ModelCfg():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.dropout = None
        self.pca_config = PCAConfig()


def print_macs(m):
    macs, params = get_model_complexity_info(m, (3, 224, 224), as_strings=True, verbose=True)
    print(macs)  # 2.11 GMac
    print(params)


def main():
    cfg = ModelCfg(1000)
    m = occam_resnet18(1000)
    m.set_use_exit_gate(True)
    m.set_return_early_exits(True)
    print_macs(m)  # 2.11 GMac

    from variable_width_resnet import resnet18
    m2 = resnet18(1000)
    print_macs(m2)  # 1.82 GMac


if __name__ == "__main__":
    main()
