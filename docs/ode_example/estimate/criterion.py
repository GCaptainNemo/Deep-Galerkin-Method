#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 22:49

import torch.autograd


class Criterion:
    def __init__(self, model, data_sampler):
        self.model = model
        self.data_sampler = data_sampler

    def loss_func(self, mse=False):
        if mse:
            x_y_sample_train = self.data_sampler.sample_x_y(all=True)
            x_y_sample_train.requires_grad = True
            x_batch_train = x_y_sample_train[:, 0].reshape(-1, 1)
            _, y_batch_train = self.model(x_batch_train)
            _ = _.detach()
            mse = 10 * torch.max((y_batch_train - x_y_sample_train[:, 1]) ** 2)
            return mse

        x_batch, x_0 = self.data_sampler.sample()
        x_batch.requires_grad = True
        cond_batch, y_batch = self.model(x_batch)
        _, y_0 = self.model(x_0)
        jacob_matrix = torch.autograd.grad(y_batch, x_batch, grad_outputs=torch.ones_like(y_batch),
                                           create_graph=True)
        dy_dx = jacob_matrix[0][:, 0].reshape(-1, 1)
        error = torch.mean((cond_batch * dy_dx - 1) ** 2) + torch.mean(y_0 ** 2)

        # ##############################################
        # train loss
        x_y_sample_train = self.data_sampler.sample_x_y()
        # x_y_sample_train.requires_grad = True
        x_batch_train = x_y_sample_train[:, 0].reshape(-1, 1)
        cond_train, y_batch_train = self.model(x_batch_train)
        jacob_matrix_train = torch.autograd.grad(y_batch_train, x_batch_train, grad_outputs=torch.ones_like(y_batch_train),
                                           create_graph=True)
        dy_dx_train = jacob_matrix_train[0][:, 0].reshape(-1, 1)

        mse = 100 * torch.mean((y_batch_train - x_y_sample_train[:, 1]) ** 2)
        train_error = mse + torch.mean((cond_train * dy_dx_train - 1) ** 2)

        # ####################################################################
        error += train_error
        return error
