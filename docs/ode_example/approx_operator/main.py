#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/28 14:30 
from docs.ode_example.approx_operator.model import *
from docs.ode_example.approx_operator.model_point import *

from docs.ode_example.approx_operator.data import *
from docs.ode_example.approx_operator.criterion import *
from docs.ode_example.approx_operator.train import *


# train_data = CustomDataset("chebyshev.pkl")
train_data = CustomDataset("gaussian_ux.pkl")

train_loader = data.DataLoader(dataset=train_data, batch_size=30, shuffle=True)
model = OperatorPointApprox(30, 10)
# model = OperatorApprox(30, 100, 30, 100)
# model = torch.load("model.pth")
# model = model.cuda()

criterion = Criterion(model)
train = Train(train_loader, criterion)

train.train(1000, 1e-3)
torch.save(model, 'model_point.pth')







