#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 22:47

import torch


class DataSampler:
    def __init__(self, train_size, x_y_observe):
        """
        :param batch_size:
        :param sample_size: boundary size
        """
        self.x_y_observe = x_y_observe
        # self.x_y_observe.requires_grad = True
        self.train_size = train_size
        self.grad = self.grad_estimate()

    def grad_estimate(self):
        num = self.x_y_observe.shape[0]
        grad = torch.zeros([num, 1])
        for i in range(1, num):
            grad[i, 0] = (self.x_y_observe[i, 1] - self.x_y_observe[i - 1, 1]) / \
                         (self.x_y_observe[i, 0] - self.x_y_observe[i - 1, 0])
        return grad

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
        # noise = torch.rand([self.train_size, 1]) * 0.1
        # x_y_sample = self.x_y_observe[index, :] + noise
        x_y_sample = self.x_y_observe[index, :]
        grad_estimate = self.grad[index, :]
        return x_y_sample, grad_estimate


