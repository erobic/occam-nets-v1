from torchvision.transforms import *
from omegaconf.listconfig import ListConfig
import logging
import torch
import torchvision.transforms.functional as tF


def _build_transformation(aug_type, aug_cfg):
    if '_' in aug_type:
        aug_type = aug_type.split("_")[0]
    if getattr(aug_cfg, 'enabled'):
        key_vals = {}
        for attr_key in aug_cfg.keys():
            if attr_key == 'enabled':
                continue
            if attr_key == 'p':
                continue  # Probability is handled through RandomApply
            curr_attr = getattr(aug_cfg, attr_key)
            if isinstance(curr_attr, ListConfig):
                curr_attr = list(curr_attr)
            key_vals[attr_key] = curr_attr

        logging.getLogger().debug(f"Adding: {aug_type}")
        if aug_type in ['brightness', 'contrast', 'saturation', 'hue']:
            cj_dict = {'brightness': 0, 'contrast': 0, 'saturation': 0, 'hue': 0, aug_type: key_vals['value']}
            aug = ColorJitter(**cj_dict)
        elif 'RandomGrayscale' in aug_type or 'RandomHorizontalFlip' in aug_type:
            key_vals['p'] = 1  # Randomness is defined by RandomApply
            aug = eval(aug_type)(**key_vals)
        else:
            aug = eval(aug_type)(**key_vals)
        # try:
        if hasattr(aug_cfg, 'p'):
            if 'joint' in aug_type.lower():
                aug = JointRandomApply([aug], p=aug_cfg.p)
            else:
                aug = RandomApply([aug], p=aug_cfg.p)
        # except:
        #     logging.getLogger().exception(f"Could not apply RandomApply to {aug_type}")
        return aug


def build_transformation_list(augmentation_cfg=None, image_size=None):
    """

    :param augmentation_cfg: Specify parameters for each augmentation type including whether or not it is enabled
    :param image_size: Final image size
    :return:
    """
    single_list = []
    joint_list = []

    if augmentation_cfg is not None:
        for aug_type in augmentation_cfg.keys():
            curr_cfg = getattr(augmentation_cfg, aug_type)
            if curr_cfg.enabled:
                t = _build_transformation(aug_type, curr_cfg)
                if 'joint' in aug_type.lower():
                    joint_list.append(t)
                else:
                    single_list.append(t)

    if image_size is not None:
        single_list.append(transforms.Resize([image_size, image_size]))
        # joint_list.append(JointResize([image_size, image_size]))
    single_list.append(transforms.ToTensor())
    return single_list, joint_list


# Creating extra augmentation classes such that image + mask are applied the same augmentations
class JointRandomHorizontalFlip(RandomHorizontalFlip):
    """
    Applies identical horizontal flips to all the images. Assumes all images have the same size.
    """

    def forward(self, images):
        _images = images
        if torch.rand(1) < self.p:
            for img_ix, img in enumerate(images):
                _images[img_ix] = tF.hflip(img)
        return _images


class JointRandomCrop(RandomCrop):
    """
    Applies identical random crop to all the images. Assumes all images have the same size.
    """

    def forward(self, images):
        _images = images
        if self.padding is not None:
            for img_ix, img in enumerate(images):
                _images[img_ix] = tF.pad(img, self.padding, self.fill, self.padding_mode)

        height, width = tF.get_image_size(images[0])
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            for img_ix, img in enumerate(_images):
                _images[img_ix] = tF.pad(img, padding, self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]

            for img_ix, img in enumerate(_images):
                _images[img_ix] = tF.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(_images[0], self.size)
        for img_ix, img in enumerate(_images):
            _images[img_ix] = tF.crop(img, i, j, h, w)
        return _images


class JointRandomResizedCrop(RandomResizedCrop):
    """
    Applies identical random resized crop transform to all the images. Assumes all images have the same size.
    """

    def forward(self, images):
        img = images[0]
        _images = images
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        for img_ix, img in enumerate(images):
            _images[img_ix] = tF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return _images


class JointRandomApply(RandomApply):
    def forward(self, images):
        _images = images
        if self.p < torch.rand(1):
            return _images
        for t in self.transforms:
            _images = t(_images)
        return _images


class JointResize(Resize):
    def forward(self, images):
        _images = []
        for img in images:
            _images.append(tF.resize(img, self.size, self.interpolation, self.max_size, self.antialias))
        return _images


class JointToTensor(ToTensor):
    def __call__(self, images):
        _images = []
        for img in images:
            _images.append(super().__call__(img))
        return _images
