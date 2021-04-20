#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/17 1:16 
import torch
import numpy as np
import pickle
te = 4
xe = 1
ye = 1
# from src.model import Net
# model = Net(2, 20)
model = torch.load('../model/net_model_new.pt')
model.eval()

t_range = np.linspace(0, te, 100, dtype=np.float64)
x_range = np.linspace(0, xe, 100, dtype=np.float64)
y_range = np.linspace(0, ye, 100, dtype=np.float64)
# data = np.meshgrid(t_range, x_range, y_range)
train_data = np.zeros((len(t_range), len(x_range), len(y_range), 4))

print(train_data.shape)
for i_t, _t in enumerate(t_range):
    for i_x, _x in enumerate(x_range):
        for i_y, _y in enumerate(y_range):
            train_data[i_t, i_x, i_y, 0] = _t
            train_data[i_t, i_x, i_y, 1] = _x
            train_data[i_t, i_x, i_y, 2] = _y
            indata = torch.Tensor(train_data[i_t, i_x, i_y, :3])
            temp = model(indata).detach().cpu().numpy()
            # print("temp = ", temp)
            train_data[i_t, i_x, i_y, 3] = temp

with open("../train_data/train_data_offset_new.pkl", "wb") as f:
    pickle.dump(train_data, f)

