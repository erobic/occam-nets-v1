import unittest

import torch
from trainers.occam_trainer import *


class TestOccamTrainer(unittest.TestCase):

    def test_cam_suppression_loss(self):
        loss_fn = CAMSuppressionLoss()
        cams = torch.randn((5, 10, 7, 7))
        gt_ys = torch.LongTensor([1, 3, 5, 0, 1])
        loss_fn(cams, gt_ys)


if __name__ == '__main__':
    unittest.main()
