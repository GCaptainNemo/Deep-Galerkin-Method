#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/27 11:25 

import torch.nn as nn
import torch


class OperatorApprox(nn.Module):
    def __init__(self, trunk_depth, trunk_width, branch_depth, branch_width):
        super(OperatorApprox, self).__init__()
        self.branch_input = nn.Linear(100, branch_width)
        self.branch_hidden = nn.Linear(branch_width, branch_width)
        self.branch_output = nn.Linear(branch_width, 10)
        self.branch_depth = branch_depth

        # ##############################################3
        self.trunk_input = nn.Linear(1, trunk_width)
        self.trunk_hidden = nn.Linear(trunk_width, trunk_width)
        self.trunk_output = nn.Linear(trunk_width, 10)
        self.trunk_depth = trunk_depth

        # ############################################
        self.bias = nn.Parameter(torch.tensor([0], dtype=torch.float32,
                                              requires_grad=True))
        self.register_parameter("bias", self.bias)


    def branch_net(self, x):
        x = self.activate(self.branch_input(x))
        for i in range(self.branch_depth):
            x = self.branch_hidden(x)
            input_ = x
            x = self.activate(x)
            x + input_
        x = self.activate(self.branch_output(x))
        return x

    def trunk_net(self, x):
        x = self.activate(self.trunk_input(x))
        for i in range(self.trunk_depth):
            x = self.trunk_hidden(x)
            input_ = x
            x = self.activate(x)
            x + input_
        x = self.trunk_output(x)
        return x

    def forward(self, x):
        # print("x.shape = ", x.shape)
        branch_input = x[:, :100]
        trunk_input = x[:, 100].reshape(-1, 1)
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        guy = torch.sum(branch_output * trunk_output, dim=1) + self.bias
        # print("bias = ", self.bias)
        # guy = branch_output @ trunk_output.t()
        # print("guy shape = ", guy.shape)
        return guy

    def activate(self, x):
        return torch.tanh(x)


# class BranchNet(nn.Module):
#     def __init__(self, trunk_depth, trunk_width):
#         super(BranchNet, self).__init__()
#         self.trunk_depth = trunk_depth
#         self.trunk_width = trunk_width
#         self.input_layer = nn.Linear(100, trunk_width)
#         self.hidden_layer = nn.Linear(trunk_width, trunk_width)
#         self.bn_layer = nn.BatchNorm1d(trunk_width)
#         self.trunk_depth = trunk_depth
#         self.output_layer = nn.Linear(trunk_width, 10)
#
#     def forward(self, x):
#         # x = [x0, x1, x2, ..., x99]
#         # print("branch net = ", x.shape)
#         x = self.activate(self.input_layer(x))
#         for i in range(self.trunk_depth):
#             x = self.hidden_layer(x)
#             input_ = x
#             x = self.activate(x)
#             x + input_
#         x = self.activate(self.output_layer(x))
#         return x
#
#     def activate(self, x):
#         return torch.tanh(x)


class TrunkNet(nn.Module):
    def __init__(self, depth, width):
        super(TrunkNet, self).__init__()
        self.depth = depth
        self.input_layer = nn.Linear(1, width)
        self.hidden_layer = nn.Linear(width, width)
        self.bn_layer = nn.BatchNorm1d(width)
        self.output_layer = nn.Linear(width, 10)

    def forward(self, x):
        # x = [u0]
        # print("trunk net = ", x.shape)
        x = self.activate(self.input_layer(x))
        for i in range(self.depth):
            x = self.hidden_layer(x)
            input_ = x
            x = self.activate(x)
            x + input_
        x = self.output_layer(x)
        return x

    def activate(self, x):
        return torch.tanh(x)


