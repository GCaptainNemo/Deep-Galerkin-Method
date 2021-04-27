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

    def loss_func(self, mse_=False):
        mse_ = False
        if mse_:
            x_y_sample_train, grad = self.data_sampler.sample_x_y()
            x_batch_train = x_y_sample_train[:, 0].reshape(-1, 1)
            y_batch_train = self.model(x_batch_train)
            mse = torch.mean((y_batch_train - x_y_sample_train[:, 1]) ** 2)
            return mse
        elif mse_ == False:
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
            return 10 * gradient_loss + init_mse
            # return gradient_loss
        else:
            # take it as piece linear function
            x_batch, y_batch, grad_batch = self.data_sampler.new_sample()
            x_batch.requires_grad = True
            y_batch_output = self.model(x_batch)
            mse = (y_batch_output - y_batch) ** 2
            jacobi = torch.autograd.grad(y_batch_output, x_batch, grad_outputs=torch.ones_like(y_batch_output), create_graph=True)
            dy_dx = jacobi[0][:, 0].reshape([-1, 1])

            grad_mse = (dy_dx - grad_batch) ** 2
            error = torch.mean(mse) + torch.mean(grad_mse)
            return error
