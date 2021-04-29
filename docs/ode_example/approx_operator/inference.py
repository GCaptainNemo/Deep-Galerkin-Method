#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/28 23:36 
import torch
import matplotlib.pyplot as plt

model = torch.load("model.pth")
model.eval()
N = 1000

u = torch.ones(N, 1) @ torch.linspace(0, 4, 100).reshape([1, -1])
y = torch.rand([N, 1]) * 4

input_ = torch.cat([u, y], dim=1)
output_ = model(input_).detach().numpy()
print(output_.shape)
plt.figure(1)
plt.scatter(y, output_, c="r", s=2)

plt.show()






