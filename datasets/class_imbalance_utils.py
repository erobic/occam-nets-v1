import torch
import numpy as np


def sample_long_tailed(class_ixs, class_imbalance_ratio, num_classes):
    """
    Returns item ixs for a new dataset with the specified imbalance ratio
    :param class_ixs:
    :param class_imbalance_ratio: (# instances in the rarest/# instance in the most frequent class)
    :param num_classes:
    :return:
    """
    if isinstance(class_ixs, list):
        class_ixs = np.asarray(class_ixs)
    num_data_per_class = get_num_data_per_class(class_ixs, class_imbalance_ratio, num_classes)
    sampled_item_ixs = []
    for class_ix in range(num_classes):
        all_item_ixs = np.where(class_ixs == class_ix)[0]
        np.random.shuffle(all_item_ixs)
        sampled_item_ixs += all_item_ixs[:num_data_per_class[class_ix]].tolist()
    return sampled_item_ixs


def get_num_data_per_class(class_ixs, class_imbalance_ratio, num_classes):
    """
    :param class_ixs: Class id for each sample
    :param class_imbalance_ratio: (# instances in the rarest/# instance in the most frequent class)
    :return: List of (item_ix, class_ix) pairs sampled at the given class imbalance ratio
    """
    max_data = len(class_ixs) / num_classes
    num_per_class = []
    for class_ix in range(num_classes):
        curr_num = max_data * (class_imbalance_ratio ** (class_ix / (num_classes - 1.0)))
        num_per_class.append(int(curr_num))
    return num_per_class


if __name__ == "__main__":
    # print(get_num_data_per_class([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], class_imbalance_ratio=1/5, num_classes=3))
    print(sample_long_tailed(np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]), class_imbalance_ratio=1/5, num_classes=3))
