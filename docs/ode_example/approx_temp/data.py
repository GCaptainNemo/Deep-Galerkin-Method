#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 22:47

import torch


class DataSampler:
    def __init__(self, train_size, x_y_observe, tx):
        """
        :param batch_size:
        :param sample_size: boundary size
        """
        self.x_y_observe = x_y_observe
        # self.x_y_observe.requires_grad = True
        self.train_size = train_size
        self.grad = self.grad_estimate()
        self.tx = tx

    def grad_estimate(self):
        num = self.x_y_observe.shape[0]
        grad = torch.zeros([num, 1])
        for i in range(1, num):
            grad[i, 0] = (self.x_y_observe[i, 1] - self.x_y_observe[i - 1, 1]) / \
                         (self.x_y_observe[i, 0] - self.x_y_observe[i - 1, 0])
        print("grad = ", grad)
        return grad

    def sample_x_y(self, all=False):
        """
        A set of observation of the ode.
        :return: x_batch, y_batch
        """
        if all:
            return self.x_y_observe, self.grad
        Num = self.x_y_observe.shape[0]
        # print("Num = ", Num)
        index = torch.randint(0, Num, [self.train_size])
        noise = torch.randn([self.train_size]) * 0.1
        x_y_sample = self.x_y_observe[index, :]
        # x_y_sample[:, 0] += noise
        # self.x_y_observe = self.x_y_observe[index, 0] + noise

        # x_y_sample = self.x_y_observe[index, :]
        grad_estimate = self.grad[index, :]
        return x_y_sample, grad_estimate

    def sample_init_point(self):
        return self.x_y_observe[0, :]

    def sample_grad(self):
        Num = self.x_y_observe.shape[0]
        # print("Num = ", Num)
        index = torch.randint(0, Num, [self.train_size])
        grad_estimate = self.grad[index, :]
        return grad_estimate

    def new_sample(self):
        Num = self.x_y_observe.shape[0]
        x_batch = torch.rand([self.train_size, 1]) * 4
        y_batch = torch.zeros([self.train_size, 1])
        grad_batch = torch.zeros([self.train_size, 1])
        # print(self.grad.shape)
        for i in range(self.train_size):
            x_index = int(x_batch[i] // self.tx)
            lamb = (x_batch[i] - self.x_y_observe[x_index, 0]) / self.tx

            y_batch[i, 0] = self.x_y_observe[x_index, 1] + \
            self.grad[x_index] * (x_batch[i] - self.x_y_observe[x_index, 0])
            if x_index < Num - 1:
                grad_batch[i, 0] = self.grad[x_index, 0] * (1 - lamb) + lamb * self.grad[x_index + 1, 0]
            else:
                grad_batch[i, 0] = self.grad[x_index, 0]
        return x_batch, y_batch, grad_batch

