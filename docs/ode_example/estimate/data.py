#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 22:47

import torch


class DataSampler:
    def __init__(self, batch_size, sample_size, train_size, x_y_observe):
        """
        :param batch_size:
        :param sample_size: boundary size
        """
        self.whole_space_size = batch_size
        self.boundary_size = sample_size
        self.x_y_observe = x_y_observe
        self.x_y_observe.requires_grad = True
        self.train_size = train_size

    def sample(self):
        x_batch = 4 * torch.rand([self.whole_space_size, 1])
        x_0 = torch.zeros([self.boundary_size, 1])
        return x_batch, x_0

    def sample_x_y(self, all=False):
        """
        A set of observation of the ode.
        :return: x_batch, y_batch
        """
        if all:
            return self.x_y_observe
        Num = self.x_y_observe.shape[0]
        # print("Num = ", Num)
        index = torch.randint(0, Num, [self.train_size])
        x_y_sample = self.x_y_observe[index, :]
        return x_y_sample


