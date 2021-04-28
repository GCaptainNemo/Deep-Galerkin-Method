#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/28 14:30 
from docs.ode_example.approx_operator.model import *
from docs.ode_example.approx_operator.data import *
from docs.ode_example.approx_operator.criterion import *
from docs.ode_example.approx_operator.train import *


train_data = CustomDataset("chebyshev.pkl")
train_loader = data.DataLoader(dataset=train_data, batch_size=10, shuffle=True)

# model = OperatorApprox(4, 40, 4, 40)
model = torch.load("model.pth")
# model = model.cuda()

criterion = Criterion(model)
train = Train(train_loader, criterion)

train.train(15000, 1e-4)
torch.save(model, 'model.pth')







