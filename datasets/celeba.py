import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from utils.data_utils import dict_collate_fn
from datasets.base_dataset import BaseDataset
from datasets.augmentation_utils import *


class CelebADataset(BaseDataset):
    """
    CelebA dataset (already cropped and centered). This code is adapted from: https://github.com/kohpangwei/group_DRO.
    """

    def __init__(self, data_dir, target_name, bias_variable_names, transform=None):
        self.data_dir = data_dir
        self.target_name = target_name
        self.bias_variable_names = bias_variable_names
        self.transform = transform

        # Read in attributes.
        col_names = ['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                     'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                     'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                     'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                     'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                     'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                     'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        self.attributes = pd.read_csv(
            os.path.join(data_dir, 'Anno', 'list_attr_celeba.txt'), sep='\s+', names=col_names, skiprows=2)

        # Split out image ids and attribute names
        self.data_dir = os.path.join(self.data_dir, 'Img', 'img_align_celeba')
        self.image_ids = self.attributes['image_id'].values
        self.attributes = self.attributes.drop(labels='image_id', axis='columns')
        self.attribute_names = self.attributes.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attribute_vals = self.attributes.values
        self.attribute_vals[self.attribute_vals == -1] = 0

        # Get the y values
        target_location = self.get_attribute_location(self.target_name)
        self.y_array = self.attribute_vals[:, target_location]
        self.num_classes = 2

        # Map the bias variables to a number 0,...,2^|confounder_idx|-1
        self.bias_variable_idx = [self.get_attribute_location(a) for a in self.bias_variable_names]
        self.num_bias_variables = len(self.bias_variable_idx)
        bias_variables = self.attribute_vals[:, self.bias_variable_idx]
        bias_variable_id = bias_variables @ np.power(2, np.arange(len(self.bias_variable_idx)))
        self.bias_variable_ixs = bias_variable_id

        # Map to groups
        # Note, we are grouping things by label and bias variable
        self.num_groups = self.num_classes * pow(2, len(self.bias_variable_idx))
        self.group_array = (self.y_array * (self.num_groups / 2) + self.bias_variable_ixs).astype('int')

        # Read in train/val/test splits
        self.split_df = pd.read_csv(
            os.path.join(data_dir, 'Eval', 'list_eval_partition.txt'), sep='\s+', names=['image_id', 'partition'])
        self.split_array = self.split_df['partition'].values
        self.features_mat = None

    def __len__(self):
        return len(self.image_ids)

    def get_attribute_location(self, attr_name):
        return self.attribute_names.get_loc(attr_name)

    def __getitem__(self, ix):
        ix = int(ix)
        y = self.y_array[ix]
        group_ix = self.group_array[ix]
        img_filename = os.path.join(self.data_dir, self.image_ids[ix])

        img = Image.open(img_filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        ret_obj = {'x': img,
                   'y': y,
                   'group_ix': group_ix,
                   'item_ix': ix,
                   'filename': img_filename,

                   }
        ret_obj['maj_min_group_ix'] = self.bias_variable_ixs[ix]

        # Add bias variables
        for bias_name in self.bias_variable_names:
            bias_val = self.attributes[bias_name].values[ix]
            ret_obj[bias_name] = bias_val
        ret_obj[self.target_name] = self.attributes[self.target_name].values[ix]

        # Add group_name
        ret_obj['group_name'] = self.group_str(group_ix)
        return ret_obj

    def group_str(self, group_idx):
        y = group_idx // (self.num_groups / self.num_classes)
        c = group_idx % (self.num_groups // self.num_classes)

        if self.target_name == 'Blond_Hair' and len(self.bias_variable_names) == 1 and self.bias_variable_names[
            0] == 'Male':
            target_names = ['Non-Blond', 'Blond']
            attr_names = ['Non-Male', 'Male']
            group_name = target_names[int(y)] + ' ' + attr_names[int(c)]
        else:
            group_name = f'{self.target_name} = {int(y)}'
            bin_str = format(int(c), f'0{self.num_bias_variables}b')[::-1]
            for attr_idx, attr_name in enumerate(self.bias_variable_names):
                group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name


def create_celeba_datasets(dataset_cfg):
    col_names = ['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                 'Bangs',
                 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                 'Wearing_Necktie', 'Young']
    data_dir = dataset_cfg.data_dir
    attrs_df = pd.read_csv(
        os.path.join(data_dir, 'Anno', 'list_attr_celeba.txt'), sep='\s+', names=col_names, skiprows=2)
    split_df = pd.read_csv(
        os.path.join(data_dir, 'Eval', 'list_eval_partition.txt'), sep='\s+', names=['image_id', 'partition'])
    split_array = split_df['partition'].values
    split_dict = {
        'Train': 0,
        'Val': 1,
        'Test': 2
    }

    split_to_dataset = {}
    for split in ['Train', 'Val', 'Test']:
        # Gather indices for current data split
        split_mask = split_array == split_dict[split]
        split_indices = np.where(split_mask)[0]

        # Filter data based on specified filter/limit
        filtered_indices_list = []
        # if filter is not None:
        #     for key in filter:
        #         key_values = attrs_df[key].values
        #         filter_mask = key_values == filter[key]
        #         filtered_indices = np.where(filter_mask)[0]
        #         filtered_indices_list.append(filtered_indices)

        final_filtered_indices = split_indices
        for curr_filtered_indices in filtered_indices_list:
            final_filtered_indices = np.intersect1d(final_filtered_indices, curr_filtered_indices)
        if split == 'Train' and dataset_cfg.train_ratio is not None:
            np.random.shuffle(final_filtered_indices)
            filter_len = int(len(final_filtered_indices) * dataset_cfg.train_ratio)
            final_filtered_indices = final_filtered_indices[0:filter_len]
            logging.getLogger().info(f"Length of dataset {len(final_filtered_indices)}")
        if split == 'Train':
            augmentation_cfg = dataset_cfg.augmentations
            transform_list = build_transformation_list(augmentation_cfg,
                                                       image_size=dataset_cfg.image_size)

        else:
            transform_list = [transforms.CenterCrop(dataset_cfg.original_image_size),
                              transforms.Resize(dataset_cfg.image_size),
                              transforms.ToTensor()
                              ]
        if dataset_cfg.normalize:
            transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform_list)
        dataset = CelebADataset(data_dir, target_name=dataset_cfg.target_name,
                                bias_variable_names=dataset_cfg.bias_variable_names,
                                transform=transform)
        dataset = Subset(dataset, final_filtered_indices)
        split_to_dataset[split] = dataset
    return split_to_dataset


def create_celeba_dataloaders(config):
    """
    Uses the train loader for training and remaining for testing
    :param config:
    :return:
    """
    split_to_datasets = create_celeba_datasets(config.dataset)
    out = {}
    out['Train'] = DataLoader(
        split_to_datasets['Train'],
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        collate_fn=dict_collate_fn()
    )
    out['Test'] = {'Train': out['Train']}
    for split in split_to_datasets:
        if split.lower() != 'train':
            out['Test'][split] = DataLoader(
                split_to_datasets[split],
                batch_size=config.dataset.batch_size,
                shuffle=False,
                num_workers=config.dataset.num_workers,
                collate_fn=dict_collate_fn()
            )
    # print(out['Train'].dataset)
    config.dataset.num_groups = out['Train'].dataset.dataset.num_groups
    return out
