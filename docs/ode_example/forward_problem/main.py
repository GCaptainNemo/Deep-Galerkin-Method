#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 23:00

from docs.ode_example.forward_problem.model import *
from docs.ode_example.forward_problem.data import *
from docs.ode_example.forward_problem.criterion import *
from docs.ode_example.forward_problem.train import *
import torch

model = SolveOde(3, 30)
# model = torch.load("model.pth")


data_sampler = DataSampler(100, 1)  # 100 data, 1 boundary data
criterion = Criterion(model, data_sampler)

train = Train(criterion)
train.train(5000, 3e-4)

torch.save(model, 'model.pth')







