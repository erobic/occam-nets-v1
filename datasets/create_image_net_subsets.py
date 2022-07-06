import os
import json
import numpy as np


def build_image_net_subsets():
    # 1%, 2%, 4%, 8%, 16%, 32%
    data_dir = '/home/robik/datasets/ImageNet1K'
    path = '/home/robik/datasets/ImageNet1K/train'
    cls_dirs = os.listdir(path)
    subset_info = {}
    for ratio in [1, 2, 4, 8, 16, 32, 100]:
        subset_info[ratio] = {}
        for cls_dir in cls_dirs:
            img_files = os.listdir(os.path.join(path, cls_dir))
            np.random.shuffle(img_files)
            subset_len = int(len(img_files) * ratio / 100)
            subset_info[ratio][cls_dir] = img_files[:subset_len]
    with open(os.path.join('datasets', 'image_net_subset.json'), 'w') as f:
        json.dump(subset_info, f)


if __name__ == "__main__":
    build_image_net_subsets()
