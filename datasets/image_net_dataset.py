import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import dict_collate_fn
import os
import numpy as np
from datasets.augmentation_utils import build_transformation_list
import torch
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def map_imagenet_classes(data_dir='/home/robik/datasets/ImageNet1K'):
    with open(os.path.join(data_dir, 'words.txt')) as f:
        ids = list(sorted(os.listdir(os.path.join(data_dir, 'val'))))
        lines = f.readlines()
        id_descr = [l.split("\t") for l in lines]
        id_to_descr = {l[0]: l[1].replace('\n', '') for l in id_descr}
        descr_to_id = {id_to_descr[k]: k for k in id_to_descr}
    ix_to_descr = {ix: id_to_descr[id] for ix, id in enumerate(ids)}
    return ix_to_descr, id_to_descr, descr_to_id


class ImageNetDataset(Dataset):
    def __init__(self, data_dir='/home/robik/datasets/ImageNet1K',
                 split='val',
                 transform=transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor()]
                 ),
                 mask_transform=None,
                 num_classes=100,
                 subset_percent=None,
                 sampling_proba=1.1,
                 logits_file=None,
                 minority_ratio=0.2):
        """

        :param data_dir:
        :param split:
        :param transform:
        :param num_classes: Orders the class ids alphabetically and loads data from the specified # of classes
        :param sampling_proba: Use a value >= 1 if you want the full set, else specify a value < 1
        """
        super(ImageNetDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.num_classes = num_classes
        self.subset_percent = subset_percent
        if self.subset_percent is not None:
            self.subset = json.load(open(os.path.join(data_dir, 'subset_info.json')))[str(subset_percent)]
        self.sampling_proba = sampling_proba
        self.transform = transform
        self.mask_transform = mask_transform
        self.logits_file = logits_file
        self.minority_ratio = minority_ratio
        if self.logits_file is not None:
            self.create_groups()
        self.load_item_map()

    def load_item_map(self):
        self.item_map = {}
        class_dirs = list(sorted(os.listdir(os.path.join(self.data_dir, 'val'))))[:self.num_classes]
        item_ix = 0
        self.class_ix_to_name, _, _ = map_imagenet_classes(data_dir=self.data_dir)
        if self.split in ['val_mask', 'test_mask']:
            _split = self.split.split('_mask')[0]
            with open(os.path.join(self.data_dir, 'localization', f"imglist_{_split}.txt")) as f:
                mask_img_files = {k.replace('\n', ''): k.replace('\n', '') for k in f.readlines()}
        for cls_ix, cls_dir in enumerate(class_dirs):
            if self.split in ['val_mask', 'test_mask']:
                _split = 'val'
            else:
                _split = self.split
            img_files = os.listdir(os.path.join(self.data_dir, _split, cls_dir))
            if self.split in ['val_mask', 'test_mask']:
                filt_img_files = []
                for i in img_files:

                    if i.replace('.JPEG', '') in mask_img_files:
                        filt_img_files.append(i)
                img_files = filt_img_files

            rands = np.random.rand((len(img_files)))
            for img_ix, img_file in enumerate(img_files):
                if self.subset_percent is not None:
                    if img_file not in self.subset[cls_dir]:
                        continue
                if self.sampling_proba is None or rands[img_ix] <= self.sampling_proba:
                    if hasattr(self, 'minority_item_ixs'):
                        is_minority = item_ix in self.minority_item_ixs
                        group_ix = (2 * cls_ix) + int(is_minority)
                        group_name = f'{cls_dir} minority={is_minority}'

                    else:
                        group_ix = cls_ix
                        group_name = cls_dir
                    self.item_map[item_ix] = {
                        'item_ix': item_ix,
                        'y': torch.LongTensor([cls_ix]),
                        'file_name': os.path.join(cls_dir, img_file),
                        'group_ix': group_ix,
                        'group_name': group_name,
                        'class_code': cls_dir,
                        'class_name': self.class_ix_to_name[int(cls_ix)]
                    }
                    for k in self.item_map[item_ix]:
                        setattr(self, f'{k}_{item_ix}', self.item_map[item_ix][k])  # To prevent memory leaks
                    item_ix += 1

    def __len__(self):
        # return 256
        return len(self.item_map)

    def __getitem__(self, index):
        # item = self.item_map[index] # Memory leak
        item = {}
        for k in ['item_ix', 'y', 'file_name', 'group_ix', 'group_name', 'class_name']:
            item[k] = getattr(self, f'{k}_{index}')

        mask = None
        if self.split in ['val_mask']:
            img_name = os.path.join(self.data_dir, 'val', item['file_name'])
            mask_name = os.path.join(self.data_dir, 'localization', 'images',
                                     item['file_name'].split('/')[1].replace('.JPEG', '.png'))
            mask = pil_loader(mask_name)
        else:
            img_name = os.path.join(self.data_dir, self.split, item['file_name'])

        img = pil_loader(img_name)

        if self.transform is not None:
            img = self.transform(img)
            if mask is not None:
                mask = self.mask_transform(mask)  # TODO: take care of it during train augmentation later

        item['x'] = img
        if mask is not None:
            item['mask'] = mask
        return item

    def create_groups(self):
        results_holder = torch.load(self.logits_file)
        ixs = torch.argsort(results_holder['losses'], descending=True)
        minority_ixs = ixs[:int(len(ixs) * self.minority_ratio)]
        self.minority_item_ixs = {}
        # sanity check -- do all classes fall under this minority?
        cls_to_minority = {}
        for ix in minority_ixs:
            y = int(results_holder['gt_labels'][ix])
            if y not in cls_to_minority:
                cls_to_minority[y] = 0
            cls_to_minority[y] += 1
            self.minority_item_ixs[int(results_holder['item_ixs'][ix])] = 1
        assert len(cls_to_minority) == self.num_classes


def create_image_net_dataset_for_split(dataset_cfg, split):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_dir = dataset_cfg.data_dir

    if 'train' in split.lower():
        augmentation_cfg = dataset_cfg.augmentations
        train_transform = build_transformation_list(augmentation_cfg, image_size=dataset_cfg.image_size)[0]
        train_transform.append(normalize)
        train_transform = transforms.Compose(train_transform)
        return ImageNetDataset(data_dir,
                               'train',
                               transform=train_transform,
                               num_classes=dataset_cfg.num_classes,
                               logits_file=dataset_cfg.logits_file,
                               subset_percent=dataset_cfg.subset_percent
                               )
    else:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]
        )
        mask_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
        )
        return ImageNetDataset(data_dir,
                               split.lower(),
                               transform=test_transform,
                               mask_transform=mask_transform,
                               num_classes=dataset_cfg.num_classes,
                               subset_percent=None
                               )


def create_image_net_datasets(dataset_cfg):
    split_to_dataset = {'val': {}, 'test': {}}
    split_to_dataset['train'] = create_image_net_dataset_for_split(dataset_cfg, 'Train')

    eval_splits = ['val_mask', 'val']  # 'val'
    for eval_split in eval_splits:
        split_to_dataset['val'][eval_split.lower()] = create_image_net_dataset_for_split(dataset_cfg, eval_split)
        split_to_dataset['test'][eval_split.lower()] = create_image_net_dataset_for_split(dataset_cfg, eval_split)

    return split_to_dataset


def create_image_net_dataloader_for_split(dataset_cfg, split):
    dataset = create_image_net_dataset_for_split(dataset_cfg, split)

    if 'train' in split.lower():
        return DataLoader(dataset,
                          batch_size=dataset_cfg.batch_size,
                          shuffle=True,
                          num_workers=dataset_cfg.num_workers,
                          collate_fn=dict_collate_fn())
    else:
        return DataLoader(dataset,
                          batch_size=dataset_cfg.batch_size,
                          shuffle=False,
                          num_workers=dataset_cfg.num_workers,
                          collate_fn=dict_collate_fn(),
                          pin_memory=True)


def create_image_net_dataloaders(config):
    config.dataset.num_classes = 1000
    config.dataset.num_groups = 1000
    split_to_datasets = create_image_net_datasets(config.dataset)
    out = {}
    out['train'] = create_image_net_dataloader_for_split(config.dataset, 'Train')
    out['val'] = {}
    out['test'] = {}
    for eval_split in split_to_datasets['test']:
        out['test'][eval_split] = create_image_net_dataloader_for_split(config.dataset, eval_split)
    for eval_split in split_to_datasets['val']:
        out['val'][eval_split] = create_image_net_dataloader_for_split(config.dataset, eval_split)
    return out


def create_image_net_100_dataloaders(config):
    config.dataset.num_classes = 100
    config.dataset.num_groups = 100
    split_to_datasets = create_image_net_datasets(config.dataset)

    out = {}
    out['Train'] = DataLoader(split_to_datasets['Train'],
                              batch_size=config.dataset.batch_size,
                              shuffle=True,
                              num_workers=config.dataset.num_workers,
                              collate_fn=dict_collate_fn())
    out['Test'] = {}
    # out['Test'] = {'Train': out['Train']}
    for eval_split in split_to_datasets['Test']:
        out['Test'][eval_split] = DataLoader(split_to_datasets['Test'][eval_split],
                                             batch_size=config.dataset.batch_size,
                                             shuffle=False,
                                             num_workers=config.dataset.num_workers,
                                             collate_fn=dict_collate_fn(),
                                             pin_memory=True)
    return out
