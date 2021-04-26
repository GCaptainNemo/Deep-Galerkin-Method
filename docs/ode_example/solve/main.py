#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 23:00 

from docs.ode_example.solve.model import *
from docs.ode_example.solve.data import *
from docs.ode_example.solve.criterion import *
from docs.ode_example.solve.train import *

# model = ApproxSolve(15, 2) # 10 layer 2 node
model = torch.load("model.pth")
data_sampler = DataSampler(100, 1)  # 100 data, 10 boundary data
criterion = Criterion(model, data_sampler)

train = Train(criterion)
train.train(5000, 1e-4)

torch.save(model, 'model.pth')







