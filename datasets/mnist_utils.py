import numpy as np
import copy
import os, json
import itertools
import cv2
from emnist import extract_training_samples, extract_test_samples
import colorsys


def map_factor_value_to_ix(factor_to_values):
    factor_value_to_ix = {}
    for factor in factor_to_values:
        if factor not in factor_value_to_ix:
            factor_value_to_ix[factor] = {}
        vals = factor_to_values[factor]
        val_ix = 0
        for val in vals:
            val = str(val)
            factor_value_to_ix[factor][val] = val_ix
            val_ix += 1
    return factor_value_to_ix


def get_non_biased_values_per_class(class_ix_to_factor_value, num_classes=10):
    """
    Assumes that each class co-occurs frequently with the factor value at the same index
    :return:
    """
    class_ix_to_non_biased_values = {}
    for class_ix in np.arange(0, num_classes):
        non_biased_vals = copy.deepcopy(class_ix_to_factor_value)
        del non_biased_vals[class_ix]  # Remove the biased value
        class_ix_to_non_biased_values[class_ix] = non_biased_vals
    return class_ix_to_non_biased_values


def get_digit_colors():
    digit_colors = [
        (200, 200, 200),
        (0, 150, 255),
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 235),
        (255, 140, 0),
        (155, 3, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 0),
    ]
    return digit_colors


def get_digit_hues():
    digit_hues = [
        0,  # red
        20,  # brownish
        40,  # orange
        60,  # yellow
        90,  # light green
        140,  # green
        175,  # cyan
        220,  # blue
        270,  # purple,
        295,  # pink
    ]
    return digit_hues


def get_digit_grayscales():
    digit_colors = np.asarray(get_texture_colors())
    print(np.mean(digit_colors, axis=1))
    #


def get_default_digit_color():
    return (255, 255, 255)


def get_letter_colors():
    return np.flip(get_digit_colors())


def apply_color(src_img, target_rgb):
    if len(src_img.shape) == 2:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    bgr = src_img.mean(axis=2)
    bgr = np.expand_dims(bgr, 2).repeat(3, axis=2)
    bgr[:, :, 0] *= target_rgb[2] / 255
    bgr[:, :, 1] *= target_rgb[1] / 255
    bgr[:, :, 2] *= target_rgb[0] / 255
    return bgr


def get_texture_colors():
    texture_colors = [(r / 1.5, g / 1.5, b / 1.5) for (r, g, b) in np.flip(get_digit_colors(), axis=0)]
    # hsv_colors = np.asarray([colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2]) for rgb in texture_colors])
    # hsv_colors[:, 1] = hsv_colors[:, 1] / 2
    # hsv_colors[:, 2] = hsv_colors[:, 2] / 2
    # texture_colors = np.asarray([colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2]) for hsv in hsv_colors])

    return texture_colors


def get_default_texture_color():
    return (0, 0, 0)


def get_digit_scales(num_cells=5, pad_cells=0):
    """
    We define scale as the fraction of the image (in terms of number of cells in horizontal/vertical dims) occupied
    by the digits.
    :param num_cells:
    :param pad_cells:
    :return:
    """
    return np.linspace(1, num_cells - pad_cells - pad_cells / 10, 10).tolist()


def get_default_digit_scale():
    return 1


def get_scale_ix_to_digit_positions(num_cells=5, pad_cells=0):
    """
    Different scales have different possible positions e.g., x1, y1 of a 3x3 digit needs to be at least (maxCellX - 3, maxCellY - 3)
    :param num_cells:
    :param pad_cells:
    :return:
    """
    scales = get_digit_scales(num_cells, pad_cells)
    scale_ix_to_digit_positions = {}
    for scale_ix, scale in enumerate(scales):
        scale_positions = []
        for row in np.arange(0, num_cells - np.ceil(scale) + 1):
            for col in np.arange(0, num_cells - np.ceil(scale) + 1):
                scale_positions.append((row, col))
        np.random.shuffle(scale_positions)
        scale_ix_to_digit_positions[scale_ix] = scale_positions
    return scale_ix_to_digit_positions


def sample_biased_digit_positions(digit_scale_ixs, scale_ix_to_positions, class_ixs, p_bias):
    """
    Returns the biased/canonical value of position for a given scale with probability p_bias, else chooses position at random
    :param digit_scale_ixs:
    :param scale_ix_to_positions: map from scale to potential positions
    :param p_bias:
    :return:
    """
    value_to_ix = {}
    max_value_to_ix = 0

    random_p = np.random.random(len(digit_scale_ixs))

    # Choose the values for each factor
    sampled_factor_ixs = []
    sampled_factors = []
    for ix, (scale_ix, cls_ix) in enumerate(zip(digit_scale_ixs, class_ixs)):
        if random_p[ix] <= p_bias[cls_ix]:
            # Assign the biased value for each factor
            sampled_factor = scale_ix_to_positions[scale_ix][0]
        else:
            # Sample uniformly from rest of the factors
            if len(scale_ix_to_positions[scale_ix]) > 1:
                arr = np.asarray(scale_ix_to_positions[scale_ix][1:])
                arr_ix = np.random.randint(0, len(arr))
                sampled_factor = arr[arr_ix]
            else:
                sampled_factor = scale_ix_to_positions[scale_ix][0]

        sampled_factors.append(sampled_factor)
        if str(sampled_factor) not in value_to_ix:
            value_to_ix[str(sampled_factor)] = max_value_to_ix
            max_value_to_ix += 1
        sampled_factor_ixs.append(value_to_ix[str(sampled_factor)])

    return sampled_factor_ixs, sampled_factors


def get_center_positions(digit_scales, num_cells=5):
    """
    Given digit scales, it returns (x,y) for top left portion of the image that ensures the object is centered
    :param digit_scales:
    :param num_cells:
    :return:
    """
    return [((num_cells - scale) // 2, (num_cells - scale) // 2) for scale in digit_scales]


def get_default_digit_position():
    return (0, 0)


def get_letters():
    # return ['a', 'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'p']
    return ['c', 'k', 'm', 'n', 'p', 'r', 'u', 'v', 'w', 'x']


def get_letter_ord():
    return [ord(c) - ord('a') for c in get_letters()]


def get_letter_ord_to_ix():
    ords = get_letter_ord()
    letter_ord_to_ix = {}
    for ix, ord in enumerate(ords):
        letter_ord_to_ix[ord] = ix
    return letter_ord_to_ix


def load_letter_ix_to_images(split):
    """
    :param split:
    :return:
    """
    # Load the images and labels
    if split == 'test':
        images, labels = extract_test_samples('letters')
    else:
        images, labels = extract_training_samples('letters')

    # Create label to images
    letter_ix_to_images = {}
    valid_letter_ords = get_letter_ord()
    letter_ord_to_ix = get_letter_ord_to_ix()
    for ix, (img, l) in enumerate(zip(images, labels)):
        # img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        label_ord = l - 1
        if label_ord in valid_letter_ords:
            letter_ix = letter_ord_to_ix[label_ord]
            if letter_ix not in letter_ix_to_images:
                letter_ix_to_images[letter_ix] = []
            letter_ix_to_images[letter_ix].append(img)

    # Perform train/val splits
    if split != 'test':
        for l in letter_ix_to_images:
            if split == 'val':
                letter_ix_to_images[l] = letter_ix_to_images[l][0:1000]
            else:
                letter_ix_to_images[l] = letter_ix_to_images[l][1000:]

    return letter_ix_to_images


def sample_conditional_biased_values(biased_value_list, biased_ixs, bias_ix_to_p_bias, avoid_ixs=None):
    """
    Samples biased value i.e., the value corresponding to biased_value_list[biased_ix] where bias_ix is the
    corresponding value specified in biased_ixs with probability p_bias

    If avoid_ixs is specified and if the biased_ix is same as avoid_ix, then it chooses something else at random

    :param biased_value_list: a map or an array from class ix to biased value based on some variable (size = num of classes)
    :param biased_ixs: Indicates the most prominent factor for each sample (size = num samples)
    :param bias_ix_to_p_bias: p_bias per bias value
    :param avoid_ixs: If biased_ix is same as avoid_ix then choose a different value at random
    :return:
    """
    if isinstance(biased_ixs, list):
        biased_ixs = np.asarray(biased_ixs)
    if avoid_ixs is not None and isinstance(avoid_ixs, list):
        avoid_ixs = np.asarray(avoid_ixs)

    all_factor_ixs = np.zeros_like(biased_ixs)
    all_factors = []
    unq_biased_ixs = np.unique(biased_ixs)

    for biased_ix in unq_biased_ixs:
        curr_ixs = np.where(biased_ixs == biased_ix)[0]
        curr_biased_ixs = biased_ixs[curr_ixs]
        curr_avoid_ixs = None
        if avoid_ixs is not None:
            curr_avoid_ixs = avoid_ixs[curr_ixs]
        sampled_factor_ixs, sampled_factors = sample_biased_values(biased_value_list, curr_biased_ixs,
                                                                   bias_ix_to_p_bias[biased_ix], curr_avoid_ixs)
        all_factor_ixs[curr_ixs] = sampled_factor_ixs

    for factor_ix in all_factor_ixs:
        all_factors.append(biased_value_list[factor_ix])

    return all_factor_ixs.tolist(), all_factors


def sample_biased_values(biased_value_list, biased_ixs, p_bias, avoid_ixs=None):
    """
    Samples biased value i.e., the value corresponding to biased_value_list[biased_ix] where bias_ix is the
    corresponding value specified in biased_ixs with probability p_bias

    If avoid_ixs is specified and if the biased_ix is same as avoid_ix, then it chooses something else at random

    :param biased_value_list: a map or an array from class ix to biased value based on some variable (size = num of classes)
    :param biased_ixs: Indicates the most prominent factor for each sample (size = num samples)
    :param p_bias:
    :param avoid_ixs: If biased_ix is same as avoid_ix then choose a different value at random
    :return:
    """
    if isinstance(biased_value_list, np.ndarray):
        biased_value_list = biased_value_list.tolist()
    value_to_ix = {}
    for cix, bv in enumerate(biased_value_list):
        value_to_ix[str(bv)] = cix

    class_ix_to_non_biased_values = get_non_biased_values_per_class(biased_value_list)

    # For a given sample, if bias_proba <= random_p, then the correlated/biased value for that factor is used
    # Else, we randomly sample from the remaining values
    random_p = np.random.random(len(biased_ixs))

    # Choose the values for each factor
    sampled_factor_ixs = []
    sampled_factors = []
    for ix, biased_ix in enumerate(biased_ixs):
        if random_p[ix] <= p_bias and (avoid_ixs is None or biased_ix != avoid_ixs[ix]):
            # Assign the biased value for each factor
            sampled_factor = biased_value_list[biased_ix]
        else:
            # Sample uniformly from rest of the factors
            rand_ix = int(np.random.choice(len(class_ix_to_non_biased_values[biased_ix]), size=1))
            sampled_factor = class_ix_to_non_biased_values[biased_ix][rand_ix]

        sampled_factors.append(sampled_factor)
        sampled_factor_ixs.append(value_to_ix[str(sampled_factor)])

    return sampled_factor_ixs, sampled_factors


def save_or_load_sampled_factors(save_dir, filename, rewrite=True, sampled_factors=None):
    """

    :param self:
    :param save_dir:
    :param rewrite:
    :param sampled_factors:
    :return:
    """
    factor_json_file = os.path.join(save_dir, f'{filename}.json')
    if not os.path.exists(factor_json_file) or rewrite:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, f'{filename}.json'), 'w') as f:
            json.dump(sampled_factors, f, indent=4, sort_keys=True)

    sampled_factors = json.load(open(factor_json_file))
    return sampled_factors


class GroupUtils():
    def __init__(self, target_name, bias_variable_names, num_classes=10, use_majority_minority_grouping=False):
        """
        :param factor_to_values:  map from name of the factor e.g., digit or digit_color to all possible values
        :param factor_value_to_ix: map from factor to value of factor to index
        :param num_classes:
        :param target_name:
        :param use_majority_minority_grouping:
        """
        self.num_classes = num_classes
        self.target_name = target_name
        self.bias_variable_names = bias_variable_names
        self.use_majority_minority_grouping = use_majority_minority_grouping
        self.group_name_to_ix = {}
        self.maj_min_group_name_to_ix = {}
        self.max_group_ix = 0
        self.max_maj_min_group_ix = 0

    def to_group_ix_and_name(self, curr_factor_to_val):
        group_name_parts = []

        # Assume that if the factor ix is same as the index of the factor val, then it is a majority group,
        # else it is a minority group
        maj_min_group_name_parts = []

        try:
            class_ix = curr_factor_to_val[self.target_name]
        except:
            print('here')

        # Go through all of the bias variables, to come up with the group name
        for ix, bias_name in enumerate(self.bias_variable_names):
            bias_val_ix = curr_factor_to_val[bias_name]
            maj_min = 'minority'
            if bias_val_ix == class_ix:
                maj_min = 'majority'  # There is no majority/minority for lbl, so we just use 'majority' for label
            group_name_parts.append(f'{bias_name}_{bias_val_ix}')
            maj_min_group_name_parts.append(f'{bias_name}_{maj_min}')

        group_name = '+'.join(group_name_parts)
        maj_min_group_name = '+'.join(maj_min_group_name_parts)
        if self.use_majority_minority_grouping:
            group_name = maj_min_group_name

        if group_name not in self.group_name_to_ix:
            self.group_name_to_ix[group_name] = self.max_group_ix
            self.max_group_ix += 1
        if maj_min_group_name not in self.maj_min_group_name_to_ix:
            self.maj_min_group_name_to_ix[maj_min_group_name] = self.max_maj_min_group_ix
            self.max_maj_min_group_ix += 1
        group_ix = self.group_name_to_ix[group_name]
        maj_min_group_ix = self.maj_min_group_name_to_ix[maj_min_group_name]
        return group_ix, group_name, maj_min_group_ix, maj_min_group_name


def preprocess_mnist(img, width=32, height=32):
    img = np.repeat(img, 3, axis=2)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) * 255
    return img


def dataset_to_xy(dataset, width=32, height=32):
    images, labels = [], []
    for img, label in dataset:
        img = np.asarray(img).transpose(1, 2, 0)
        img = preprocess_mnist(img, width, height)
        images.append(img)
        labels.append(label)
    return np.asarray(images), np.asarray(labels)


def count_groups(test):
    # file = '/hdd/robik/biased_mnist/full/test.json'
    # test = json.load(open(file))

    def get_grp_name(factor):
        return f'dig_{factor["digit"]}_col_{factor["digit_color_ix"]}_scale_{factor["digit_scale_ix"]}_' \
               f'texture_{factor["texture_ix"]}_txt_color_{factor["texture_color_ix"]}'

    factor_cnts = {}
    for f in test:
        grp_name = get_grp_name(f)
        if grp_name not in factor_cnts:
            factor_cnts[grp_name] = 0
        factor_cnts[grp_name] += 1

    print(json.dumps(factor_cnts, indent=4, sort_keys=True))

    print(len(factor_cnts))


def perturb_saturation_and_values(hues, min_saturation=50, min_value=70):
    saturations = np.random.uniform(low=min_saturation, high=100, size=(len(hues)))
    values = np.random.uniform(low=min_value, high=100, size=(len(hues)))
    rgbs = np.asarray(
        [colorsys.hsv_to_rgb(h / 360, s / 100, v / 100) for (h, s, v) in zip(hues, saturations, values)]) * 255.
    return rgbs


if __name__ == "__main__":
    get_digit_grayscales()
