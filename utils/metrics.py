import torch


class Accuracy():
    """
    Tracks accuracy per class + group
    """

    def __init__(self, top_k=[1, 5]):
        self.top_k = top_k
        self.reset()

    def reset(self):
        self.correct_dict = {}
        self.total_dict = {}

    def update(self, pred_scores, gt_ys, class_names=None, group_names=None):
        sorted_ys = torch.argsort(pred_scores, dim=-1, descending=True)
        for ix in range(len(sorted_ys)):
            cls_name = str(gt_ys[ix]) if class_names is None else class_names[ix]
            grp_name = cls_name if group_names is None else group_names[ix]
            k_to_ys = {k: sorted_ys[ix][:k] for k in self.top_k}
            self._update_one(k_to_ys, gt_ys[ix], cls_name, 'class')
            self._update_one(k_to_ys, gt_ys[ix], grp_name, 'group')

    def _update_one(self, k_to_ys, gt_y, name, grp_type):
        name = str(name)

        if grp_type not in self.total_dict:
            self.total_dict[grp_type] = {}
            self.correct_dict[grp_type] = {}

        if name not in self.total_dict[grp_type]:
            self.total_dict[grp_type][name] = 0
            self.correct_dict[grp_type][name] = {k: 0 for k in k_to_ys}

        self.total_dict[grp_type][name] += 1

        for k in k_to_ys:
            ys = [int(y) for y in k_to_ys[k]]
            if int(gt_y) in ys:
                self.correct_dict[grp_type][name][k] += 1

    def get_per_group_accuracy(self, group_type='group', factor=1):
        assert group_type in self.total_dict
        per_group_accuracy = {}
        for group_name in self.correct_dict[group_type]:
            if group_name not in per_group_accuracy:
                per_group_accuracy[group_name] = {}
            for k in self.correct_dict[group_type][group_name]:
                per_group_accuracy[group_name][k] \
                    = self.correct_dict[group_type][group_name][k] / self.total_dict[group_type][group_name] * factor
        return per_group_accuracy

    def get_mean_per_group_accuracy(self, group_type='group', factor=1, topK=None):
        per_group_accuracy = self.get_per_group_accuracy(group_type)
        mpg = {}
        for k in self.top_k:
            total_acc, total_num = 0, 0
            for group_name in per_group_accuracy:
                total_acc += per_group_accuracy[group_name][k]
                total_num += 1
            mpg[k] = total_acc / total_num * factor
        if topK is not None:
            return mpg[topK]
        else:
            return mpg

    def get_accuracy(self, group_type='class', factor=1):
        acc_dict = {}
        for k in self.top_k:
            correct, total = 0, 0
            for grp_name in self.total_dict[group_type]:
                correct += self.correct_dict[group_type][grp_name][k]
                total += self.total_dict[group_type][grp_name]
            acc_dict[k] = correct / total * factor
        return acc_dict

    def summary(self, factor=100):
        obj = {}
        # obj['accuracy'] = self.get_accuracy('class', factor)
        acc = self.get_accuracy('class', factor)
        mpc = self.get_mean_per_group_accuracy('class', factor)
        mpg = self.get_mean_per_group_accuracy('group', factor)
        for k in acc:
            obj[f'Top {k} Acc'] = acc[k]
            # obj[f'Top {k} MPC'] = mpc[k]
            # obj[f'Top {k} MPG'] = mpg[k]
        return obj

    def detailed(self, factor=100):
        return self.summary(factor)
        # obj = {}
        # for group_type in self.total_dict:
        #     obj[group_type] = {
        #         'total': self.total_dict[group_type],
        #         'correct': self.correct_dict[group_type],
        #         'accuracy': self.get_accuracy(group_type, factor),
        #         'MPG': self.get_mean_per_group_accuracy(group_type, factor),
        #         'per_group': self.get_per_group_accuracy(group_type, factor)
        #     }
        # return obj


class ExitPercent():
    def __init__(self):
        self.total = 0
        self.exited = 0

    def update(self, exited):
        if exited:
            self.exited += 1
        self.total += 1

    def get_exit_pct(self, factor=100):
        # if self.total == 0:
        #     return 0
        return self.exited / self.total * factor

    def summary(self):
        return {
            'Exit%': self.get_exit_pct()
        }
