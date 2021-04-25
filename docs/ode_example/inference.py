#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/17 1:16

import torch
import matplotlib.pyplot as plt


model = torch.load('model.pth')
model.eval()
xe = 4
x_batch = torch.linspace(0, xe, 400, dtype=torch.float32).reshape(-1, 1)
# x_range = np.linspace(0, xe, 400, dtype=np.float64)
y_batch = model(x_batch).numpy()
print(y_batch)
# with open("../train_data/train_data_ones.pkl", "wb") as f:
#     pickle.dump(train_data, f)

plt.figure(1)
plt.plot(x_batch.numpy(), y_batch)
plt.show()
