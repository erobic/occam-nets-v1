import torch
from torch import nn


def instance_std(x, eps=1e-5):
    N, C, H, W = x.size()
    x1 = x.reshape(N * C, -1)
    var = x1.var(dim=-1, keepdim=True) + eps
    return var.sqrt().reshape(N, C, 1, 1)


def x_times_std(x, groups, eps=1e-5):
    N, C, H, W = x.size()
    x1 = x.reshape(N, groups, -1)
    var = (x1.var(dim=-1, keepdim=True) + eps).reshape(N, groups, -1)
    return (x1 * torch.rsqrt(var)).reshape(N, C, H, W)


def get_num_groups(num_channels, max_size=32):
    i = 0
    grp_size = 1
    while True:
        _grp_size = 2 ** i
        i += 1
        if num_channels % _grp_size == 0:
            grp_size = _grp_size
        else:
            break

    return min(grp_size, max_size)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (
                sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i))
        )


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def instance_std(x, eps=1e-5):
    var = torch.var(x, dim=(2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)


def group_std(x, groups, eps=1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim=(2, 3, 4), keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))


def group_std_1d(x, groups, eps=1e-5):
    N, C = x.size()
    x = torch.reshape(x, (N, groups, C // groups))
    var = torch.var(x, dim=(2), keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C))


class EvoNormND(nn.Module):
    # https://raw.githubusercontent.com/digantamisra98/EvoNorm/master/models/evonorm2d.py
    def __init__(
            self,
            num_features,
            groups=None,
            ndims=4,
            non_linear=True,
            version="S0",
            efficient=False,
            affine=True,
            momentum=0.9,
            eps=1e-5,
            training=True,
            use_v1=True
    ):
        super(EvoNormND, self).__init__()
        if groups is None:
            groups = num_features
        self.ndims = ndims
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.efficient = efficient
        if self.version == "S0":
            self.swish = MemoryEfficientSwish()
        self.use_v1 = use_v1
        self.groups = groups
        self.eps = eps
        if self.version not in ["B0", "S0"]:
            raise ValueError("Invalid EvoNorm version")
        self.num_features = num_features
        self.affine = affine

        if self.affine:
            if self.ndims == 4:
                self.gamma = nn.Parameter(torch.ones(1, self.num_features, 1, 1))
                self.beta = nn.Parameter(torch.zeros(1, self.num_features, 1, 1))
                if self.non_linear and self.use_v1:
                    self.v = nn.Parameter(torch.ones(1, self.num_features, 1, 1))
            elif self.ndims == 2:
                self.gamma = nn.Parameter(torch.ones(1, self.num_features))
                self.beta = nn.Parameter(torch.zeros(1, self.num_features))
                if self.non_linear and self.use_v1:
                    self.v = nn.Parameter(torch.ones(1, self.num_features))

        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
            self.register_buffer("v", None)
        if self.ndims == 4:
            self.register_buffer("running_var", torch.ones(1, self.num_features, 1, 1))
        elif self.ndims == 2:
            self.register_buffer("running_var", torch.ones(1, self.num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.dim() != self.ndims:
            raise ValueError(f"expected {self.ndims}D input (got {x.dim()}D input)")

    def forward(self, x):
        self._check_input_dim(x)
        if self.version == "S0":
            if self.non_linear:
                if not self.efficient:
                    # Original Swish Implementation, however memory intensive.
                    if self.use_v1:
                        num = x * torch.sigmoid(self.v * x)
                    else:
                        num = x * torch.sigmoid(x)
                else:
                    # Experimental Memory Efficient Variant of Swish
                    num = self.swish(x)
                if self.ndims == 4:
                    std_fn = group_std
                elif self.ndims == 2:
                    std_fn = group_std_1d
                return (num / std_fn(x, groups=self.groups, eps=self.eps) * self.gamma + self.beta)
            else:
                return x * self.gamma + self.beta
        if self.version == "B0":
            if self.training:
                if self.ndims == 4:
                    var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
                else:
                    var = torch.var(x, dim=(0), unbiased=False, keepdim=True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                if self.use_v1:
                    den = torch.max((var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps))
                else:
                    den = torch.max((var + self.eps).sqrt(), x + instance_std(x, eps=self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta


class EvoNorm1D_S0(EvoNormND):
    def __init__(self, num_features):
        super().__init__(num_features, ndims=2)


class EvoNorm2D_S0(EvoNormND):
    def __init__(self, num_features):
        super().__init__(num_features, ndims=4)


class EvoNorm1D_S1(EvoNormND):
    def __init__(self, num_features):
        super().__init__(num_features, use_v1=False, ndims=2)


class EvoNorm2D_S1(EvoNormND):
    def __init__(self, num_features):
        super().__init__(num_features, use_v1=False, ndims=4)
