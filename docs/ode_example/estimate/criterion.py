#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 22:49

import torch.autograd


class Criterion:
    def __init__(self, model, data_sampler):
        self.model = model
        self.data_sampler = data_sampler

    def loss_func(self, option="temp"):
        if option == "temp":
            x_y_sample_train, grad_train = self.data_sampler.sample_x_y()

            x_batch_train = x_y_sample_train[:, 0].reshape(-1, 1)
            x_batch_train.requires_grad = True
            _, y_batch_train = self.model(x_batch_train)
            mse_temp_train = 10 * torch.max((y_batch_train - x_y_sample_train[:, 1]) ** 2)
            # ####################################################################

            jacobi_train = torch.autograd.grad(y_batch_train, x_batch_train, grad_outputs=torch.ones_like(y_batch_train), create_graph=True)
            dy_dx = jacobi_train[0][:, 0].reshape(-1, 1)
            grad_error = 1000 * torch.mean((dy_dx - grad_train) ** 2)
            return mse_temp_train + grad_error
        elif option == "cond":
            # sample from the whole space
            x_batch, x_0 = self.data_sampler.sample()
            x_batch.requires_grad = True
            cond_batch, y_batch = self.model(x_batch)
            _, y_0 = self.model(x_0)
            jacob_matrix = torch.autograd.grad(y_batch, x_batch, grad_outputs=torch.ones_like(y_batch),
                                               create_graph=True)
            dy_dx = jacob_matrix[0][:, 0].reshape(-1, 1)
            dy_dx = dy_dx.detach()
            # fix dy_dx optimize cond
            whole_error = torch.mean((cond_batch * dy_dx - 1) ** 2) + torch.mean(y_0 ** 2)
            return whole_error
        elif option == "both":
            # sample from the whole space
            x_batch, x_0 = self.data_sampler.sample()
            x_batch.requires_grad = True
            cond_batch, y_batch = self.model(x_batch)
            _, y_0 = self.model(x_0)
            jacob_matrix = torch.autograd.grad(y_batch, x_batch, grad_outputs=torch.ones_like(y_batch),
                                               create_graph=True)
            dy_dx = jacob_matrix[0][:, 0].reshape(-1, 1)
            whole_error = torch.mean((cond_batch * dy_dx - 1) ** 2) + torch.mean(y_0 ** 2)

            # sample from the train set
            x_y_sample_train, grad = self.data_sampler.sample_x_y()
            x_batch_train = x_y_sample_train[:, 0].reshape(-1, 1)
            x_batch_train.requires_grad = True
            cond_train, y_batch_train = self.model(x_batch_train)
            jacob_matrix_train = torch.autograd.grad(y_batch_train, x_batch_train, grad_outputs=torch.ones_like(y_batch_train),
                                               create_graph=True)
            dy_dx_train = jacob_matrix_train[0][:, 0].reshape(-1, 1)
            temp_mse = torch.mean((y_batch_train - x_y_sample_train[:, 1]) ** 2)
            train_error = temp_mse + 100 * torch.mean((dy_dx_train - grad) ** 2)
            # train_error = temp_mse

            # ####################################################################
            error = whole_error + train_error
            return error
