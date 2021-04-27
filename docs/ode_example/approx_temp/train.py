#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 22:58

import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import numpy as np
from IPython import display


class Train:
    def __init__(self, criterion):
        self.criterion = criterion
        self.errors = []
        self.x_batch = torch.linspace(0, 4, 400, dtype=torch.float32).reshape(-1, 1)

    def train(self, epoch, lr):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        optimizer = optim.Adam(self.criterion.model.parameters(), lr)
        # optimizer = optim.Adam(self.criterion.bias_model.parameters(), lr)

        avg_loss = 0
        plt.figure()
        plt.ion()  # 打开交互式绘图interactive
        train_x = self.criterion.data_sampler.x_y_observe[:, 0].detach().numpy()
        train_y = self.criterion.data_sampler.x_y_observe[:, 1].detach().numpy()

        real_x = train_x
        real_y = train_y
        # real_k = np.array([1, 1])
        for e in range(epoch):
            optimizer.zero_grad()
            # if e % 1000 < 600:
            #     loss = self.criterion.loss_func(mse=True)
            # else:
            #     loss = self.criterion.loss_func()
            loss = self.criterion.loss_func(mse_=True)
            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            if e % 50 == 49:
                loss = avg_loss / 50
                print("Step {} - lr {} -  loss: {}".format(e, lr, loss))
                avg_loss = 0
                error = self.criterion.loss_func(mse_=True)
                self.errors.append(error.detach())
                y_batch = self.criterion.model(self.x_batch).detach().numpy()
                # 清除原有图像
                plt.cla()
                plt.plot(self.x_batch.numpy(), y_batch, label='估计y(x)')
                plt.plot(real_x, real_y, c="y", label="真实y(x)")
                plt.scatter(train_x, train_y, c="r", s=4)
                plt.title("loss = {}".format(loss))
                plt.legend()
                plt.pause(0.1)
                display.clear_output(wait=True)  # 每次显示完图以后删除，达到显示动图的效果
        plt.show()

    def get_errors(self):
        return self.errors


