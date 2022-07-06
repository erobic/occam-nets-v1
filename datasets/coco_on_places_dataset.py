import logging
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from datasets.mnist_utils import *
from datasets.texture_utils import *
from utils.data_utils import dict_collate_fn
from datasets.base_dataset import BaseDataset
import h5py
from datasets.augmentation_utils import *
import torch
from PIL import Image
import torchvision.transforms.functional as tF

CLASSES = [
    'boat',
    'airplane',
    'truck',
    'dog',
    'zebra',
    'horse',
    'bird',
    'train',
    'bus'
]


class CocoOnPlacesDataset(BaseDataset):
    def __init__(self, data_dir='/home/robik/datasets/coco_on_places',
                 split='valsgtest',
                 transform=transforms.Compose([
                     transforms.Resize(64),
                     transforms.ToTensor()
                 ]),
                 joint_transform=None,
                 num_classes=9
                 ):
        """

        :param data_dir:
        :param split: Possible values: train, idtest, validtest, sgtest, valsgtest, oodtest, valoodtest, anotest
        :param transform: applies only to image
        :param joint_transform: applies to both image and mask
        :param num_classes:
        """
        super(CocoOnPlacesDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.num_classes = num_classes
        self.transform = transform
        self.joint_transform = joint_transform
        h5_file = h5py.File(os.path.join(self.data_dir, f'{self.split}.h5py'))

        self.xs = np.asarray(h5_file['images'])
        if 'masks' in h5_file:
            self.masks = np.asarray(h5_file['masks'])
        if 'y' in h5_file:
            self.ys = np.asarray(h5_file['y'])
        else:
            self.ys = None
        if 'g' in h5_file:
            self.gs = np.asarray(h5_file['g'])
        else:
            self.gs = np.asarray([0] * len(self.xs))
        h5_file.close()
        self.class_ix_to_name = {ix: cls for ix, cls in enumerate(CLASSES)}

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        if self.ys is not None:
            y = self.ys[index]
            class_name = CLASSES[y]
        else:
            y = 0
            class_name = 'unknown'
        g = self.gs[index]
        group_ix = 2 * y + g  # Group by class + majority/minority group

        item = {
            'y': torch.LongTensor([y]),
            'maj_min_group_ix': g,  # 0 =majority, 1 = minority
            'group_ix': group_ix,
            'item_ix': index,
            'class_name': class_name
        }
        if g == 0:
            group_name = f"{class_name} majority"
        else:
            group_name = f"{class_name} minority"
        item['group_name'] = group_name

        # Handle image + mask
        img = self.xs[index]
        img = np.transpose(img, axes=(1, 2, 0))
        img = Image.fromarray((img * 255).astype(np.uint8), 'RGB')

        img_n_mask = [img]
        if hasattr(self, 'masks'):
            m = self.masks[index]
            m = np.transpose(m, axes=(1, 2, 0))
            m = Image.fromarray((m * 255).astype(np.uint8), 'RGB')
            img_n_mask.append(m)

        if self.joint_transform is not None:
            img_n_mask = self.joint_transform(img_n_mask)
        if self.transform is not None:
            img_n_mask[0] = self.transform(img_n_mask[0])

        item['x'] = img_n_mask[0]
        if hasattr(self, 'masks'):
            item['mask'] = img_n_mask[1]
            item['mask'] = tF.to_tensor(item['mask'])

        return item


def create_coco_on_places_dataset_for_split(dataset_cfg, split):
    if 'train' in split.lower():
        augmentation_cfg = dataset_cfg.augmentations
        train_list, joint_list = build_transformation_list(augmentation_cfg, image_size=dataset_cfg.image_size)
        single_transform, joint_transform = transforms.Compose(train_list), transforms.Compose(joint_list)
    else:
        test_list, joint_list = build_transformation_list(image_size=dataset_cfg.image_size)
        single_transform, joint_transform = transforms.Compose(test_list), transforms.Compose(joint_list)
    return CocoOnPlacesDataset(dataset_cfg.data_dir,
                               split,
                               transform=single_transform,
                               joint_transform=joint_transform,
                               num_classes=dataset_cfg.num_classes)


def create_coco_on_places_datasets(dataset_cfg):
    split_to_dataset = {'Test': {}}
    eval_splits = ['train', 'idtest', 'validtest', 'oodtest', 'valoodtest', 'sgtest', 'valsgtest', 'anotest']
    eval_aliases = ['Train', 'In Distribution', 'validtest', 'oodtest', 'valoodtest', 'Test', 'Val', 'Anomaly Detection']
    for eval_split, eval_alias in zip(eval_splits, eval_aliases):
        split_to_dataset['Test'][eval_alias] = create_coco_on_places_dataset_for_split(dataset_cfg, eval_split)
    split_to_dataset['Train'] = create_coco_on_places_dataset_for_split(dataset_cfg, 'train')

    return split_to_dataset


def create_coco_on_places_dataloader_for_split(dataset_cfg, split):
    dataset = create_coco_on_places_dataset_for_split(dataset_cfg, split)
    if 'train' in split.lower():
        return DataLoader(dataset,
                          batch_size=dataset_cfg.batch_size,
                          shuffle=True,
                          num_workers=dataset_cfg.num_workers,
                          collate_fn=dict_collate_fn(),
                          drop_last=True)
    else:
        return DataLoader(dataset,
                          batch_size=dataset_cfg.batch_size,
                          shuffle=False,
                          num_workers=dataset_cfg.num_workers,
                          collate_fn=dict_collate_fn(),
                          drop_last=True)


def create_coco_on_places_dataloaders(config):
    config.dataset.num_groups = 18  # Bias-aligned vs bias-conflicting for each of the 9 classes
    out = {'val': {}, 'test': {}}
    out['train'] = create_coco_on_places_dataloader_for_split(config.dataset, 'train')

    eval_splits = ['train', 'validtest', 'valoodtest', 'valsgtest']
    for split in eval_splits:
        out['val'][split] = create_coco_on_places_dataloader_for_split(config.dataset, split)

    test_splits = ['idtest', 'oodtest', 'sgtest']
    for split in test_splits:
        out['test'][split] = create_coco_on_places_dataloader_for_split(config.dataset, split)

    return out

def create_coco_on_places_with_mask_dataloaders(config):
    return create_coco_on_places_dataloaders(config)