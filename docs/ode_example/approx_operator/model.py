#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/27 11:25 

import torch.nn as nn
import torch


class ApproxOperator(nn.Module):
    def __init__(self, trunk_depth, trunk_width, branch_depth, branch_width):
        super(ApproxOperator, self).__init__()
        self.trunk_depth = trunk_depth
        self.trunk_width = trunk_width
        self.branch_depth = branch_depth
        self.branch_width = branch_width

    def forward(self):
        ...

    def activate(self, x):
        return torch.tanh(x)





