#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 23:00

from docs.ode_example.approx_temp.model import *
from docs.ode_example.approx_temp.data import *
from docs.ode_example.approx_temp.criterion import *
from docs.ode_example.approx_temp.train import *
import torch

# 3 layer 20 node

# model = ApproxTemp(3, 20)
model = torch.load("model.pth")

x = torch.linspace(0, 4, 100, dtype=torch.float32).reshape(-1, 1)
tx = 4 / 100

observe_x_y = torch.cat([x, x ** 3], dim=1)
# observe_x_y = torch.cat([x, x], dim=1)

# print("observe_x_y.shape = ", observe_x_y.shape)

data_sampler = DataSampler(30, observe_x_y, tx)  # 100 data, 10 boundary data

# bias_model = Bias()
criterion = Criterion(model, data_sampler)

train = Train(criterion)
train.train(5000, 1e-4)

torch.save(model, 'model.pth')







