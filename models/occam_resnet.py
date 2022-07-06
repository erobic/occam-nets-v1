from models.occam_lib import *
from models.variable_width_resnet import VariableWidthResNet, BasicBlock, Bottleneck
import math


class OccamResNet(VariableWidthResNet):
    def __init__(
            self,
            block,
            layers,
            width=64,
            initial_kernel_size=7,
            initial_stride=2,
            initial_padding=3,
            use_initial_max_pooling=True,
            # Exits
            multi_exit_type=MultiExitModule,
            exits_kwargs=None
    ) -> None:
        """
        Adds multiple exits to DenseNet
        :param width:
        :param exits_kwargs: all the parameters needed to create the exits

        """
        super().__init__(block=block,
                         layers=layers,
                         width=width,
                         initial_kernel_size=initial_kernel_size,
                         initial_stride=initial_stride,
                         initial_padding=initial_padding,
                         use_initial_max_pooling=use_initial_max_pooling)
        self.exits_cfg = exits_kwargs
        del self.fc

        multi_exit = multi_exit_type(**exits_kwargs)
        for i in range(0, 4):
            multi_exit.build_and_add_exit(getattr(self, f'layer{i + 1}')[-1].out_dims)
        self.multi_exit = multi_exit
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y=None):
        block_num_to_exit_in = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_initial_max_pooling:
            x = self.maxpool(x)

        for i in range(0, 4):
            x = getattr(self, f'layer{i + 1}')(x)
            block_num_to_exit_in[i] = x

        return self.multi_exit(block_num_to_exit_in, y=y)

    def get_multi_exit(self):
        return self.multi_exit


def occam_resnet18_img64(num_classes, width=58, multi_exit_type=MultiExitModule, exits_kwargs={}):
    if 'exit_out_dims' not in exits_kwargs:
        exits_kwargs['exit_out_dims'] = num_classes
    return OccamResNet(block=BasicBlock,
                       layers=[2, 2, 2, 2],
                       width=width,
                       initial_kernel_size=3,
                       initial_stride=1,
                       initial_padding=1,
                       use_initial_max_pooling=False,
                       multi_exit_type=multi_exit_type,
                       exits_kwargs=exits_kwargs)


def occam_resnet18(num_classes, width=58, multi_exit_type=MultiExitModule, exits_kwargs={}):
    if 'exit_out_dims' not in exits_kwargs:
        exits_kwargs['exit_out_dims'] = num_classes
    return OccamResNet(block=BasicBlock,
                       layers=[2, 2, 2, 2],
                       width=width,
                       multi_exit_type=multi_exit_type,
                       exits_kwargs=exits_kwargs)


if __name__ == "__main__":
    m = occam_resnet18_img64(20)
    print(m)
    x = torch.rand((5, 3, 224, 224))
    out = m(x)
    print(out.keys())
