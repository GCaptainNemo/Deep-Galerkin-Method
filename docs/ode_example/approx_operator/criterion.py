#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/28 14:29 

import torch


class Criterion:
    def __init__(self, model):
        self.model = model

    def loss_func(self, data, point=True):
        # # dim = [batch_num, 10]
        if point:
            input_data = data[:, :201]
            guy = data[:, -1].reshape([-1, 1])
            # print(input_data.requires_grad)
            estimate_guy = self.model(input_data)
            mse = torch.mean((estimate_guy - guy) ** 2)

            # ##################################################
            batch_num = data.shape[0]
            x_y_ = data[:, :200]
            x_hat = torch.rand([100, 1])
            # N x 1 x 10
            branch_output = torch.unsqueeze(self.model.branch_net(x_y_), dim=1)
            # 1 x 100 x 10
            trunk_output = torch.unsqueeze(self.model.trunk_net(x_hat), dim=0)
            # N x 100
            guy = torch.sum(branch_output * trunk_output, dim=2) + self.model.bias
            y = torch.ones([batch_num, 1]) @ x_hat.t()
            x_hat_y_hat = torch.cat([y, guy], dim=1)
            batch_x = data[:, :100]

            # N x 1 x 10
            branch_output_2 = self.model.branch_net(x_hat_y_hat).reshape([batch_num, 1, 10])
            # N x 100 x 10
            trunk_output_2 = self.model.trunk_net(batch_x.reshape([-1, 1])).reshape([batch_num, 100, 10])
            # N x 100
            y_estimate = torch.sum(branch_output_2 * trunk_output_2, dim=2) + self.model.bias
            y_ = data[:, 100:200]
            idempotent_error = torch.mean((y_estimate - y_) ** 2)
            total_error = 0.1 * idempotent_error + mse
            # print("total_error = ", total_error)
        return total_error

