import torch.nn as nn
from utils.metrics import Accuracy, ExitPercent
from utils.cam_utils import *
import os


class ExitDataTypes:
    EXIT_IN = 'exit_in'
    EXIT_HID = 'exit_hid'
    EXIT_OUT = 'exit_out'
    IS_CORRECT = 'is_correct'
    LOGITS = 'logits'
    EXIT_SCORE = 'exit_score'


def build_non_linearity(non_linearity_type, num_features):
    return non_linearity_type()


class Conv2(nn.Module):
    def __init__(self, in_features, hid_features, out_features, norm_type=nn.BatchNorm2d, non_linearity_type=nn.ReLU,
                 groups=1, conv_type=nn.Conv2d, kernel_size=3, stride=1):
        super(Conv2, self).__init__()
        self.conv1 = conv_type(in_channels=in_features, out_channels=hid_features, kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               groups=groups)
        self.norm1 = norm_type(hid_features)
        self.non_linear1 = build_non_linearity(non_linearity_type, hid_features)
        self.conv2 = nn.Conv2d(in_channels=hid_features, out_channels=out_features, kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               groups=groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.non_linear1(x)
        x = self.conv2(x)
        return x


class SimpleGate(nn.Module):
    def __init__(self, in_dims, hid_dims=16, output_dims=1, non_linearity_type=nn.ReLU, norm_type=nn.BatchNorm1d):
        super(SimpleGate, self).__init__()
        self.net = nn.Sequential(
            self.get_linearity_type()(in_dims, hid_dims),
            norm_type(hid_dims),
            build_non_linearity(non_linearity_type, hid_dims),
            nn.Linear(hid_dims, output_dims)
        )

    def get_linearity_type(self):
        return nn.Linear

    def forward(self, x):
        if len(x.shape) > 2:
            x = F.adaptive_avg_pool2d(x, 1).squeeze()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = self.net(x)
        x = torch.sigmoid(x).squeeze()
        return x


def get_early_exit_ixs(exit_ix_to_exit_probas):
    """
    Performs early exit based on the predicted probabilities

    :param exit_ix_to_exit_probas: A dict from exit id to gates. Has to be arranged from the earliest to the latest exit.
    :return:
    """
    # By default, exit from the final exit
    final_exit_ix = list(exit_ix_to_exit_probas.keys())[-1]
    early_exit_ixs = torch.ones_like(exit_ix_to_exit_probas[0]) * final_exit_ix
    has_exited = torch.zeros_like(exit_ix_to_exit_probas[0])

    for exit_ix in exit_ix_to_exit_probas:
        exit_probas = exit_ix_to_exit_probas[exit_ix]
        use_next_exit = (exit_probas < 0.5).int()
        early_exit_ixs = torch.where(((1 - use_next_exit) * (1 - has_exited)).bool(),
                                     torch.ones_like(exit_ix_to_exit_probas[0]) * exit_ix,
                                     early_exit_ixs)
        has_exited = torch.where((1 - use_next_exit).bool(),
                                 torch.ones_like(has_exited),
                                 has_exited)
    if len(early_exit_ixs.shape) == 0:
        early_exit_ixs = early_exit_ixs.unsqueeze(0)
    return early_exit_ixs


def get_early_exit_values(exit_ix_to_values, early_exit_ixs, clone=True):
    """
    Gather values corresponding to the specified early exit ixs
    :param exit_ix_to_values:
    :param early_exit_ixs:
    :param clone:
    :return:
    """
    earliest_exit_id = list(exit_ix_to_values.keys())[0]
    early_exit_values = exit_ix_to_values[earliest_exit_id]
    if clone:
        early_exit_values = early_exit_values.clone()

    for exit_ix in exit_ix_to_values:
        curr_ixs = torch.where(early_exit_ixs == exit_ix)[0]
        early_exit_values[curr_ixs] = exit_ix_to_values[exit_ix][curr_ixs]
    return early_exit_values


class ExitModule(nn.Module):
    """
    Exit Module consists of some conv layers followed by CAM and a gate to decide whether or not to exit
    """

    def __init__(self, in_dims, hid_dims, out_dims, cam_hid_dims=None,
                 scale_factor=1,
                 groups=1,
                 kernel_size=3,
                 stride=None,
                 initial_conv_type=Conv2,
                 conv_bias=False,
                 conv_type=nn.Conv2d,
                 norm_type=nn.BatchNorm2d,
                 non_linearity_type=nn.ReLU,
                 gate_type=SimpleGate,
                 gate_norm_type=nn.BatchNorm1d,
                 gate_non_linearity_type=nn.ReLU,
                 ):
        super(ExitModule, self).__init__()
        self.in_dims = in_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        if cam_hid_dims is None:
            cam_hid_dims = self.hid_dims
        self.cam_hid_dims = cam_hid_dims
        self.set_use_gate(True)
        self.initial_conv_type = initial_conv_type
        self.conv_bias = conv_bias
        self.conv_type = conv_type
        self.scale_factor = scale_factor
        self.groups = groups
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size // 2
        self.stride = stride
        self.norm_type = norm_type
        self.non_linearity_type = non_linearity_type
        self.gate_type = gate_type
        self.gate_norm_type = gate_norm_type
        self.gate_non_linearity_type = gate_non_linearity_type
        self.build_network()

    def build_network(self):
        self.convs = self.initial_conv_type(self.in_dims,
                                            self.hid_dims,
                                            self.cam_hid_dims,
                                            norm_type=self.norm_type,
                                            non_linearity_type=self.non_linearity_type,
                                            conv_type=self.conv_type,
                                            kernel_size=self.kernel_size,
                                            stride=self.stride)
        self.non_linearity = build_non_linearity(self.non_linearity_type, self.cam_hid_dims)
        self.cam = nn.Conv2d(
            in_channels=self.cam_hid_dims,
            out_channels=self.out_dims, kernel_size=1, padding=0)

        if self.use_gate:
            self.gate = self.gate_type(self.cam_hid_dims,
                                       norm_type=self.gate_norm_type,
                                       non_linearity_type=self.gate_non_linearity_type)

    def forward(self, x, y=None):
        """
        Returns CAM, logits and gate
        :param x:
        :return: Returns CAM, logits and gate
        """
        out = {}
        out[ExitDataTypes.EXIT_IN] = x
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, align_corners=False, mode='bilinear')

        x = self.convs(x)
        x = self.non_linearity(x)
        cam_in = x

        out['cam_in'] = cam_in
        cam = self.cam(cam_in)  # Class activation maps before pooling
        out['cam'] = cam

        out['logits'] = F.adaptive_avg_pool2d(cam, (1)).squeeze()
        if self.use_gate:
            gate_out = self.gate(x.detach())
            out['gates'] = gate_out
        return out

    def set_use_gate(self, use_gate):
        self.use_gate = use_gate


class MultiExitModule(nn.Module):
    """
    Holds multiple exits
    It passes intermediate representations through those exits to gather CAMs/predictions
    """

    def __init__(
            self,
            detached_exit_ixs=[0],
            exit_out_dims=None,
            exit_block_nums=[0, 1, 2, 3],
            exit_type=ExitModule,
            exit_gate_type=SimpleGate,
            exit_initial_conv_type=Conv2,
            exit_hid_dims=[None, None, None, None],
            exit_width_factors=[1 / 4, 1 / 4, 1 / 4, 1 / 4],
            cam_width_factors=[1, 1, 1, 1],
            exit_scale_factors=[1, 1, 1, 1],
            exit_kernel_sizes=[3, 3, 3, 3],
            exit_strides=[None] * 4,
            inference_earliest_exit_ix=1,
            downsample_factors_for_scores=[1 / 8, 1 / 4, 1 / 2, 1]
    ) -> None:
        """
        Adds multiple exits to DenseNet
        :param detached_exit_ixs: Exit ixs whose gradients should not flow into the trunk
        :param exit_out_dims: e.g., # of classes
        :param exit_block_nums: Blocks where the exits are attached (EfficientNets have 9 blocks (0-8))
        :param exit_type: Class of the exit that performs predictions
        :param exit_gate_type: Class of exit gate that decides whether or not to terminate a sample
        :param exit_initial_conv_type: Initial layer of the exit
        :param exit_width_factors:
        :param cam_width_factors:
        :param exit_scale_factors:
        :param inference_earliest_exit_ix: The first exit to use for inference (default=1 i.e., E.0 is not used for inference)

        """
        super().__init__()
        self.detached_exit_ixs = detached_exit_ixs
        self.exit_out_dims = exit_out_dims
        self.exit_block_nums = exit_block_nums
        self.exit_type = exit_type
        self.exit_gate_type = exit_gate_type
        self.exit_initial_conv_type = exit_initial_conv_type
        self.exit_hid_dims = exit_hid_dims
        self.exit_width_factors = exit_width_factors
        self.cam_width_factors = cam_width_factors
        self.exit_scale_factors = exit_scale_factors
        self.exit_kernel_sizes = exit_kernel_sizes
        self.exit_strides = exit_strides
        self.inference_earliest_exit_ix = inference_earliest_exit_ix
        self.downsample_factors_for_scores = downsample_factors_for_scores
        self.set_use_exit_gate(True)
        self.set_return_early_exits(False)
        self.exits = []

    def build_and_add_exit(self, in_dims):
        exit_ix = len(self.exits)
        _hid_dims = self.exit_hid_dims[exit_ix]
        if _hid_dims is None:
            _hid_dims = int(in_dims * self.exit_width_factors[exit_ix])
        exit = self.exit_type(
            in_dims=in_dims,
            out_dims=self.exit_out_dims,
            hid_dims=_hid_dims,
            cam_hid_dims=int(in_dims * self.cam_width_factors[exit_ix]),
            kernel_size=self.exit_kernel_sizes[exit_ix],
            stride=self.exit_strides[exit_ix],
            scale_factor=self.exit_scale_factors[exit_ix]
        )
        if hasattr(exit, 'set_downsample_factor'):
            exit.set_downsample_factor(self.downsample_factors_for_scores[exit_ix])
        self.exits.append(exit)
        self.exits = nn.ModuleList(self.exits)

    def get_exit_block_nums(self):
        return self.exit_block_nums

    def set_use_exit_gate(self, use_exit_gate):
        self.use_exit_gate = use_exit_gate

    def set_return_early_exits(self, return_early_exits):
        self.return_early_exits = return_early_exits

    def forward(self, block_num_to_exit_in, y=None, exit_strategy=None):
        exit_outs = {}
        exit_ix = 0
        for block_num in block_num_to_exit_in:
            if block_num in self.exit_block_nums:
                exit_in = block_num_to_exit_in[block_num]
                if exit_ix in self.detached_exit_ixs:
                    exit_in = exit_in.detach()
                exit_out = self.exits[exit_ix](exit_in, y=y)
                for k in exit_out:
                    exit_outs[f"E={exit_ix}, {k}"] = exit_out[k]
                exit_ix += 1

        if exit_strategy == 'combine':
            pass
        else:
            self.get_early_exits(exit_outs)
        return exit_outs

    def get_combined_logits(self, exit_outs):
        for exit_ix in range(len(self.exit_block_nums)):
            # if self.inference_earliest_exit_ix is not None and exit_ix < self.inference_earliest_exit_ix:
            #     continue
            exit_name = f"E={exit_ix}"

    def get_early_exits(self, exit_outs):
        exit_num = 0

        # Gather exit probabilities
        exit_num_to_logits, exit_num_to_exit_probas, exit_num_to_name = {}, {}, {}
        for exit_ix in range(len(self.exit_block_nums)):
            if self.inference_earliest_exit_ix is not None and exit_ix < self.inference_earliest_exit_ix:
                continue
            exit_name = f"E={exit_ix}"
            exit_num_to_name[exit_num] = exit_name
            exit_num_to_logits[exit_num] = exit_outs[f"{exit_name}, logits"]  # .detach().cpu()
            gate_key = f"{exit_name}, gates"
            if gate_key in exit_outs:
                exit_num_to_exit_probas[exit_num] = exit_outs[gate_key]
            exit_num += 1

        # Gather names and predictions for early exits
        early_exit_ixs = get_early_exit_ixs(exit_num_to_exit_probas)
        early_exit_logits = get_early_exit_values(exit_num_to_logits, early_exit_ixs)
        early_exit_names = [exit_num_to_name[int(ix)] for ix in early_exit_ixs]
        exit_outs['early_exit_names'] = early_exit_names
        exit_outs['E=early, logits'] = early_exit_logits
        _, _, final_h, final_w = exit_outs[f'E={len(self.exit_block_nums) - 1}, cam'].shape
        exit_outs['E=early, cam'] = get_early_exit_cams(exit_outs, final_h, final_w)
        return exit_outs

    def get_exit_names(self):
        names = [f'E={exit_ix}' for exit_ix in range(len(self.exit_block_nums))]
        names.append('E=early')
        return names


class MultiExitStats:
    def __init__(self):
        self.exit_ix_to_stats = {}

    def __call__(self, num_exits, exit_outs, gt_ys, class_names=None, group_names=None):
        for exit_ix in range(num_exits):
            if exit_ix not in self.exit_ix_to_stats:
                self.exit_ix_to_stats[exit_ix] = {
                    'accuracy': Accuracy(),
                    'early_exit%': ExitPercent()
                }
            logits_key = f'E={exit_ix}, logits'
            logits = exit_outs[logits_key]
            # Accuracy on all the samples
            self.exit_ix_to_stats[exit_ix]['accuracy'].update(logits, gt_ys, class_names, group_names)

            for ee_name in exit_outs['early_exit_names']:
                ee_ix = int(ee_name.split('E=')[1])
                self.exit_ix_to_stats[exit_ix]['early_exit%'].update(int(ee_ix) == int(exit_ix))

    def summary(self, prefix=''):
        exit_to_summary = {}
        for exit_ix in self.exit_ix_to_stats:
            for k in self.exit_ix_to_stats[exit_ix]:
                for k2 in self.exit_ix_to_stats[exit_ix][k].summary():
                    exit_to_summary[f"{prefix}E={exit_ix} {k2}"] = self.exit_ix_to_stats[exit_ix][k].summary()[k2]
        return exit_to_summary


class SimilarityExitModule(ExitModule):
    def __init__(self, similarity_fn=cosine_similarity, top_k=3, top_k_type='max', layer='exit_in',
                 **kwargs):
        super(SimilarityExitModule, self).__init__(**kwargs)
        self.top_k = top_k
        self.top_k_type = top_k_type
        self.similarity_fn = similarity_fn
        self.layer = layer
        self.set_downsample_factor(1)

    def set_downsample_factor(self, downsample_factor):
        """
        Feature maps are downsampled by this factor before computing similarity scores
        :param downsample_factor:
        :return:
        """
        self.downsample_factor = downsample_factor

    def forward_and_get_top_k_cells(self, x, y=None):
        """
        Gets top scoring cells from CAM (1D location in a flattened vector)
        :param x:
        :param y:
        :return:
        """
        out = super().forward(x)

        # Step 1: Get CAMs of either the GT or the highest scoring classes
        cams = out['cam']  # B x C x H x W
        hid = out[self.layer]
        _, _, hid_h, hid_w = hid.shape
        tar_h, tar_w = int(hid_h * self.downsample_factor), int(hid_w * self.downsample_factor)
        hid = interpolate(hid, tar_h, tar_w)

        classes = y if self.top_k_type == 'gt' else out['logits'].argmax(dim=-1)
        class_cams = get_class_cams_for_occam_nets(cams, classes)  # B x HW
        class_cams = interpolate(class_cams.unsqueeze(1), tar_h, tar_w).reshape(len(out['logits']), -1)

        # Step 2: Get the highest scoring cells as reference cells
        top_k_ixs = torch.argsort(class_cams, dim=1, descending=True)[:, :self.top_k]  # B x top_k

        out['ref_top_k_ixs'] = top_k_ixs
        out['ref_hid'] = hid

        return hid, top_k_ixs, out

    def calc_similarity(self, hid, top_k_ixs):
        """
        Computes similarity between all the hidden cells and the top_k cells
        :param hid: Hidden features
        :param top_k_ixs: locations for top_k cells (assuming a flattened vector)
        :return:
        """
        flat_hid = hid.reshape(hid.shape[0], hid.shape[1], -1)  # B x dims x hw
        flat_hid_top_k = torch.gather(flat_hid, dim=2, index=top_k_ixs.unsqueeze(1).repeat(1, flat_hid.shape[1], 1))
        similarity = self.similarity_fn(flat_hid, flat_hid_top_k)  # total_cells x top_k
        return similarity

    def forward(self, x, y=None):
        # Forward through the exit and obtain top_k cell locations with the highest CAM scores
        hid, top_k_ixs, out = self.forward_and_get_top_k_cells(x, y)

        # Compute similarity between all the feature map cells and the top activated cells
        out['ref_mask_scores'] = self.calc_similarity(hid, top_k_ixs).reshape(len(x),
                                                                              out['ref_hid'].shape[2],
                                                                              out['ref_hid'].shape[3])
        return out


class CosineSimilarityExitModule(SimilarityExitModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.similarity_fn = cosine_similarity


class ThresholdedCosineSimilarityExitModule(SimilarityExitModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.similarity_fn = thresholded_cosine_similarity


def visualize_reference_masks(img, sim_scores, top_k_cells, mask_h, mask_w, save_path):
    """

    :param img: Original image (3 x H x W)
    :param sim_scores: total score for each cell (shape=mask_h * mask_w)
    :param top_k_cells: Locations of flattened spatial cells
    :param mask_h, mask_w = spatial dims of reference mask
    :return:
    """
    img = (img - img.min()) / (img.max() - img.min())
    mask_top_k_cells = torch.zeros(mask_h * mask_w).to(img.device)
    mask_top_k_cells[top_k_cells] = 1
    mask_top_k_cells = mask_top_k_cells.reshape(mask_h, mask_w)
    hm = compute_heatmap(img, mask_top_k_cells)
    imwrite(os.path.join(save_path + "_top_k_cells.jpg"), hm)

    score_hm = compute_heatmap(img, sim_scores.reshape(mask_h, mask_w))
    imwrite(os.path.join(save_path + "_similarity.jpg"), score_hm)
