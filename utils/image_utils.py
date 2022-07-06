import numpy as np
import torch
import math
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def to_opponent_color_space(x):
    """
    Extracts grayscale, R-G and Y-B ch
    :param x:
    :return:
    """
    x2 = torch.zeros_like(x)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = r + g
    x2[:, 0] = (r + g + b) / 3  # grayscale
    x2[:, 1] = (y - b)  # Yellow - blue
    x2[:, 2] = (r - g)  # Red - Green

    return x2


def extract_patches(image_batch, patch_dim, stride, max_patches=None):
    """
    Copied and modified from: https://www.programcreek.com/python/?code=conscienceli%2FIterNet%2FIterNet-master%2Futils%2Fcrop_prediction.py

    :param image_batch: B x H x W x C
    :param patch_dim:
    :param stride:
    :return:
    """

    patch_height = patch_dim
    patch_width = patch_dim
    stride_height = stride
    stride_width = stride

    assert (len(image_batch.shape) == 4)  # 4D arrays
    img_h = image_batch.shape[1]  # height of the full image
    img_w = image_batch.shape[2]  # width of the full image

    # assert ((img_h - patch_height) % stride_height == 0 and (img_w - patch_width) % stride_width == 0)
    __max_num_patches = ((img_h - patch_height) // stride_height + 1) * (
            (img_w - patch_width) // stride_width + 1)  # // --> division between integers
    if max_patches is None:
        num_patches_per_img = __max_num_patches
    else:
        num_patches_per_img = max_patches
        assert max_patches <= __max_num_patches, f"Cannot compute {max_patches}, since the max # of patches is {__max_num_patches}"

    total_num_patches = num_patches_per_img * image_batch.shape[0]
    patches = np.empty((total_num_patches, patch_height, patch_width, image_batch.shape[3]))
    if max_patches is None and patch_dim == 1 and stride == 1:
        for img_ix, img in enumerate(image_batch):
            patches[img_ix * len(img.flatten()):(img_ix + 1) * len(img.flatten())] = img.flatten()
        return patches
    num_patches = 0  # iter over the total number of patches (N_patches)

    for i in range(image_batch.shape[0]):  # loop over the full images
        candidate_ix = 0
        h_limit = (img_h - patch_height) // stride_height + 1
        w_limit = (img_w - patch_width) // stride_width + 1
        if max_patches is not None:
            selected_ixs = np.random.choice(np.arange(0, h_limit * w_limit), size=max_patches, replace=False)

        for h in range(h_limit):
            for w in range(w_limit):
                if max_patches is None or candidate_ix in selected_ixs:
                    patch = image_batch[i, h * stride_height:(h * stride_height) + patch_height,
                            w * stride_width:(w * stride_width) + patch_width, :]
                    patches[num_patches] = patch
                    num_patches += 1  # total
                candidate_ix += 1

    assert (num_patches == total_num_patches)
    return patches


def extract_non_overlapping_patches(image_batch, patch_dim):
    return extract_patches(image_batch, patch_dim, patch_dim)


# def extract_non_overlapping_patches(image_batch, patch_dim):
#     patches = []
#     for img in image_batch:
#         assert img.shape[0] % patch_dim == 0
#         __patches = divide_img_blocks(img, img.shape[0] // patch_dim)
#
#         __patches = np.reshape(__patches, (-1, __patches.shape[2], __patches.shape[3], __patches.shape[4]))
#         patches.append(__patches)
#     a = np.concatenate(patches, axis=0)
#     return a


def extract_random_patches(image_batch, patch_dim, max_patches):
    return extract_patches(image_batch, patch_dim, 1, max_patches)


def divide_img_blocks(img, n_blocks):
    # Credit: https://stackoverflow.com/a/67197232/1122681
    horizontal = np.array_split(img, n_blocks)
    splitted_img = [np.array_split(block, n_blocks, axis=1) for block in horizontal]
    return np.asarray(splitted_img)


def pad_for_patches(img, patch_dim):
    H, W, _ = img.shape
    new_H = math.ceil(H / patch_dim) * patch_dim
    new_W = math.ceil(W / patch_dim) * patch_dim
    if new_H != H or new_W != W:
        return pad(img, new_H, new_W)
    else:
        return img


def pad(img, h, w):
    # https://ai-pool.com/d/padding-images-with-numpy
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(
        np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))


def identity(images):
    return images


def rotate(images, angle=30):
    return F.rotate(images, angle)


def crop(images, scale=1.2):
    h, w = images.shape[2], images.shape[3]
    new_h, new_w = int(scale * h), int(scale * w)
    images = F.resize(images, (new_h, new_w))
    return F.center_crop(images, (h, w))


def horizontal_flip(images):
    return F.hflip(images)


def color_jitter(images):
    return transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0)(images)


def add_transformations(images, transformation_types=['identity', 'rotate', 'crop', 'horizontal_flip', 'color_jitter']):
    """
    Appends transformed versions
    Number of images is multiplied by len(transformation_types)
    Assumes that the images are between 0 and 1
    :param images: B x C x H x W
    :param transformation_types
    :return: transformed_images. First len(images) entries belong to the first transform, second len(images) to the second etc...
    """
    transformed_images = torch.zeros(
        (len(transformation_types), images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
    for trans_ix, trans_type in enumerate(transformation_types):
        transformed_images[trans_ix] = eval(trans_type)(images)
    transformed_images = transformed_images.reshape((len(transformation_types) * images.shape[0],
                                                     images.shape[1],
                                                     images.shape[2],
                                                     images.shape[3]))
    return transformed_images
