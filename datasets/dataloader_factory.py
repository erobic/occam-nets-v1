from datasets.biased_mnist_dataset import create_biased_mnist_dataloaders, create_biased_mnist_dataloader_for_split
from datasets.coco_on_places_dataset import create_coco_on_places_dataloaders, \
    create_coco_on_places_dataloader_for_split, create_coco_on_places_with_mask_dataloaders
from datasets.image_net_dataset import create_image_net_dataloaders, create_image_net_100_dataloaders, \
    create_image_net_dataloader_for_split
from datasets.BAR_dataset import create_BAR_dataloaders
import logging
import torch
from torch.utils.data import DataLoader
import numpy as np


def build_balanced_loader(dataloader, balanced_sampling_attributes=['group_ix'], balanced_sampling_gamma=1,
                          replacement=True):
    """
    Weighs samples based on the group frequency (groups formed using balanced_sampling_attributes)
    :param dataloader: Original data loader
    :param balanced_sampling_attributes: Attributes to use for grouping and balancing
    :param balanced_sampling_gamma: Balancing weight = 1/n^(gamma)
    :param replacement:
    :return:
    """
    logger = logging.getLogger()
    all_group_names = []

    # Count frequencies for all groups of attributes to balance,
    # and assign each sample to a group, so that we can compute its sampling weight later on
    group_name_to_count = {}
    for batch in dataloader:
        batch_group_names = []
        for ix, _ in enumerate(batch['y']):
            group_name = ""
            for attr in balanced_sampling_attributes:
                group_name += f"{attr}_{batch[attr][ix]}_"
            batch_group_names.append(group_name)

        for group_name in batch_group_names:
            if group_name not in group_name_to_count:
                group_name_to_count[group_name] = 0
            group_name_to_count[group_name] += 1
            all_group_names.append(group_name)

    # Create the balanced loader
    weights = []
    for val in all_group_names:
        weights.append(1 / group_name_to_count[val] ** balanced_sampling_gamma)
    logging.getLogger().info(
        f'len(weights) = {len(weights)} max wt = {np.max(weights)} min wt = {np.min(weights)}')
    weighted_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights),
                                                              replacement=replacement)
    balanced_dataloader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=weighted_sampler,
                                     num_workers=dataloader.num_workers, collate_fn=dataloader.collate_fn)
    logger.info(
        f"Created balanced loader for {len(weights)} samples of dataset size {len(dataloader.dataset)} using attributes: {balanced_sampling_attributes}")
    return balanced_dataloader


def build_dataloader_for_split(cfg, split):
    """
    Builds load for the specified split of the data loader
    :param cfg:
    :param split:
    :return:
    """
    return eval(f"create_{cfg.dataset.name}_dataloader_for_split")(cfg.dataset, split)


def build_dataloaders(cfg):
    dataset_name = cfg.dataset.name
    loaders = eval(f"create_{dataset_name}_dataloaders")(cfg)

    if cfg.dataset.sampling_type == 'balanced':
        assert cfg.dataset.sampling_attributes is not None
        loaders['Train'] = build_balanced_loader(loaders['Train'],
                                                 cfg.dataset.sampling_attributes,
                                                 balanced_sampling_gamma=cfg.dataset.sampling_gamma,
                                                 replacement=True)

    return loaders
