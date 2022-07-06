import os
import cv2
import numpy as np


def get_textures():
    return [
        'grass',
        'bark',
        'fabric',
        'plain',
        'wood',
        'brick',
        'wall',
        'waves',
        'noisy',
        'plain'
    ]


def get_default_texture():
    return 'plain'


def load_texture_images(textures_dir, scale=None):
    """
    Returns a map from filename to texture image
    :param textures_dir:
    :return:
    """
    textures = get_textures()
    texture_images = {}

    for texture in textures:
        if texture not in texture_images:
            texture_images[texture] = []
        img_files = os.listdir(os.path.join(textures_dir, texture))
        for img_file in img_files:
            img = cv2.imread(os.path.join(textures_dir, texture, img_file))
            if scale is not None and scale != 1:
                img = cv2.resize(img, (img.shape[0] * scale, img.shape[1] * scale))
            texture_images[texture].append(img)
    return texture_images


def sample_texture_crops(num_samples, split, target_img_dim, texture_img_dim=1024, train_portion=2,
                         val_portion=1,
                         test_portion=1,
                         texture_img_scale=1):
    """
    Generate random numbers to select texture files and random crops within those texture files
    For ease, we assume that all full textures have dimensions of texture_img_dim x texture_img_dim

    :param texture_ixs:
    :param split:
    :param target_img_dim:
    :return:
    """
    # Select files at random
    # For each texture we choose a random value between 0 and 1 that will determine the actual image to be used
    texture_file_rand = np.random.random(num_samples)

    # For each image, we sample a random crop
    # For this we assume that the image has at least 1024 pixels in row/column.
    # We reserve first half (vertically) for train and 1/4th for val and remaining 1/4th for test
    column_width = texture_img_dim // (train_portion + val_portion + test_portion)
    if split == 'train':
        min_x1, max_x1 = 0, column_width * texture_img_scale - target_img_dim
    elif split == 'val':
        min_x1, max_x1 = column_width * texture_img_scale - target_img_dim, 2 * column_width * texture_img_scale - target_img_dim
    elif split == 'test':
        min_x1, max_x1 = 2 * column_width * texture_img_scale - target_img_dim, texture_img_dim * texture_img_scale - target_img_dim
    min_y1, max_y1 = 0, texture_img_dim * texture_img_scale - target_img_dim

    # Now create random crops
    x1s = np.random.randint(min_x1, max_x1, num_samples)
    x2s = x1s + target_img_dim
    y1s = np.random.randint(min_y1, max_y1, num_samples)
    y2s = y1s + target_img_dim
    return texture_file_rand, x1s, x2s, y1s, y2s
