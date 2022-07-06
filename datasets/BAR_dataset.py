import logging
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
from datasets.mnist_utils import *
from datasets.texture_utils import *
from utils.data_utils import dict_collate_fn
from datasets.class_imbalance_utils import *
from datasets.base_dataset import BaseDataset
from utils.image_utils import pil_loader

from datasets.augmentation_utils import *


class BARDataset(BaseDataset):
    def __init__(self, data_dir, split, transform=None, logits_file=None, minority_ratio=0.2):
        super(BARDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.num_classes = 10
        self.classes = []
        self.transform = transform
        self.images_dir = os.path.join(self.data_dir, self.split)
        self.logits_dir = logits_file
        self.minority_ratio = minority_ratio
        if self.logits_dir is not None:
            self.create_groups()

        self.prepare_dataset()

    def __len__(self):
        return len(self.ys)

    def prepare_dataset(self):
        filenames = list(sorted(os.listdir(self.images_dir)))
        self.unq_class_names = ['climbing', 'diving', 'fishing', 'pole vaulting', 'racing', 'throwing']
        self.ys = []
        self.file_names = []
        self.class_names = []
        self.class_ids = []
        self.item_ixs = []
        self.group_ixs = []

        for item_ix, fname in enumerate(filenames):
            class_name = fname.split('_')[0]
            self.ys.append(self.unq_class_names.index(class_name))
            self.class_names.append(class_name)
            self.item_ixs.append(item_ix)
            if hasattr(self, 'minority_item_ixs'):
                is_minority = item_ix in self.minority_item_ixs
                self.group_ixs.append(is_minority)  # 0 = minority i.e., has high loss, 1 = majority
            else:
                self.group_ixs.append(item_ix % 2)  # Dummy odd/even group
            self.file_names.append(fname)

    def create_groups(self):
        results_holder = torch.load(self.logits_dir)
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
        # self.minority_ixs = minority_ixs

    def __getitem__(self, index):
        fname = self.file_names[index]
        # img = pil_loader(os.path.join(self.images_dir, fname))
        img = Image.open(os.path.join(self.images_dir, fname)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        item_data = {}
        item_data['item_ix'] = index
        item_data['x'] = img
        item_data['y'] = self.ys[index]
        item_data['class_name'] = self.class_names[index]
        item_data['class_group_name'] = self.class_names[index]
        item_data['group_ix'] = self.group_ixs[index]
        item_data['file_name'] = self.file_names[index]
        return item_data


def create_BAR_datasets(cfg):
    split_to_dataset = {'Test': {}}
    train_transform = build_transformation_list(cfg.augmentations, image_size=cfg.image_size)
    train_transform = transforms.Compose(train_transform)
    train_set = BARDataset(cfg.data_dir, 'train', transform=train_transform, logits_file=cfg.logits_file)
    num_groups = len(train_set.class_names) * 2
    split_to_dataset['Train'] = train_set
    test_transform = transforms.Compose(build_transformation_list(image_size=cfg.image_size))
    test_set = BARDataset(cfg.data_dir, 'test', transform=test_transform)
    split_to_dataset['Test']['Test'] = test_set

    cfg.num_groups = num_groups

    return split_to_dataset


def create_BAR_dataloaders(config):
    ds_cfg = config.dataset
    split_to_dataset = create_BAR_datasets(ds_cfg)
    logging.getLogger().info(f"Setting the num_groups to {ds_cfg.num_groups}")
    out = {}
    out['Train'] = DataLoader(
        split_to_dataset['Train'],
        batch_size=ds_cfg.batch_size,
        shuffle=True,
        num_workers=ds_cfg.num_workers,
        collate_fn=dict_collate_fn()
    )
    out['Test'] = {'Train': out['Train']}
    for eval_split in split_to_dataset['Test']:
        out['Test'][eval_split] = DataLoader(
            split_to_dataset['Test'][eval_split],
            batch_size=ds_cfg.batch_size,
            shuffle=False,
            num_workers=ds_cfg.num_workers,
            collate_fn=dict_collate_fn()
        )
    return out
