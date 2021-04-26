#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 22:49

import torch.autograd


class Criterion:
    def __init__(self, model, data_sampler):
        self.model = model
        self.data_sampler = data_sampler
        # self.bias_model = bias_model

    def loss_func(self, mse=False):
        if mse:
            x_y_sample_train, grad = self.data_sampler.sample_x_y()
            x_batch_train = x_y_sample_train[:, 0].reshape(-1, 1)
            # x_batch_train.requires_grad =
            y_batch_train = self.model(x_batch_train)
            # y_batch_train = self.bias_model(y_batch_train)
            mse = 10 * torch.mean((y_batch_train - x_y_sample_train[:, 1]) ** 2)
            return mse
        else:
            # grad = self.data_sampler.sample_grad()
            initial = self.data_sampler.sample_init_point()
            init_y = self.model(initial[0].reshape([-1, 1]))
            init_mse = (init_y - initial[1]) ** 2

            x_y_sample_train, grad = self.data_sampler.sample_x_y()
            x_batch_train = x_y_sample_train[:, 0].reshape(-1, 1)
            x_batch_train.requires_grad = True
            y_batch_train = self.model(x_batch_train)
            # mse = torch.mean((y_batch_train - x_y_sample_train[:, 1]) ** 2)
            jacobi_matrix = torch.autograd.grad(y_batch_train, x_batch_train,
                                                grad_outputs=torch.ones_like(y_batch_train),
                                                create_graph=True)
            dy_dx_train = jacobi_matrix[0][:, 0].reshape(-1, 1)
            gradient_loss = torch.mean((dy_dx_train - grad) ** 2)
            return gradient_loss + 0.1 * init_mse