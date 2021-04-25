#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 22:47 

import torch


class DataSampler:
    def __init__(self, batch_size, sample_size):
        """
        :param batch_size:
        :param sample_size: boundary size
        """
        self.batch_size = batch_size
        self.sample_size = sample_size

    def sample(self):
        x_batch = 4 * torch.rand([self.batch_size, 1])
        x_0 = torch.zeros([self.sample_size, 1])
        return x_batch, x_0
