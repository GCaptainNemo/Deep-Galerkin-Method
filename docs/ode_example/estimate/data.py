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
        self.train_size = train_size
        self.grad = self.grad_estimate()

    def grad_estimate(self):
        num = self.x_y_observe.shape[0]
        grad = torch.zeros([num, 1])
        for i in range(1, num):
            grad[i, 0] = (self.x_y_observe[i, 1] - self.x_y_observe[i - 1, 1]) / \
                         (self.x_y_observe[i, 0] - self.x_y_observe[i - 1, 0])
        return grad

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
            return self.x_y_observe, self.grad
        batch_num = self.x_y_observe.shape[0]
        index = torch.randint(0, batch_num, [self.train_size])
        x_y_sample = self.x_y_observe[index, :]
        # x_y_sample = self.x_y_observe[index, :]
        grad_estimate = self.grad[index, :]
        return x_y_sample, grad_estimate


