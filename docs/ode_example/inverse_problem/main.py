#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 23:00

from docs.ode_example.inverse_problem.model import *
from docs.ode_example.inverse_problem.data import *
from docs.ode_example.inverse_problem.criterion import *
from docs.ode_example.inverse_problem.train import *
import torch

# 3 layer 15 node(cond, temp)
model = EstimateCond(30, 3, 30, 3)
# model = torch.load("model.pth")

x = torch.linspace(0, 4, 50, dtype=torch.float32).reshape(-1, 1)
observe_x_y = torch.cat([x, x], dim=1)
# print("observe_x_y.shape = ", observe_x_y.shape)

data_sampler = DataSampler(100, 1, 30, observe_x_y)  # 100 data, 10 boundary data
criterion = Criterion(model, data_sampler)

train = Train(criterion)
train.train(5000, 3e-4)

torch.save(model, 'model.pth')







