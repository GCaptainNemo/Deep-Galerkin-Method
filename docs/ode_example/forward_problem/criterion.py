#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 22:49

import torch.autograd


class Criterion:
    def __init__(self, model, data_sampler):
        self.model = model
        self.data_sampler = data_sampler

    def loss_func(self):
        # sample from the whole space
        x_batch, x_0 = self.data_sampler.sample()
        x_batch.requires_grad = True
        y_batch = self.model(x_batch)
        y_0 = self.model(x_0)
        jacob_matrix = torch.autograd.grad(y_batch, x_batch,
                                           grad_outputs=torch.ones_like(y_batch),
                                           create_graph=True)
        dy_dx = jacob_matrix[0][:, 0].reshape(-1, 1)
        # ode loss + initial loss
        error = torch.mean((dy_dx - 1) ** 2) + torch.mean(y_0 ** 2)
        return error
