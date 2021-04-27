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
        avg_loss = 0
        plt.figure()
        plt.ion()  # 打开交互式绘图interactive
        # real_x = np.array([0, 4])
        real_x = np.linspace(0, 4, 100)
        real_y = real_x ** 2 / 2
        # real_y = np.array([0, 4])
        for e in range(epoch):
            optimizer.zero_grad()
            loss = self.criterion.loss_func()
            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            if e % 50 == 49:
                loss = avg_loss / 50
                print("Step {} - lr {} -  loss: {}".format(e, lr, loss))
                avg_loss = 0
                error = self.criterion.loss_func()
                self.errors.append(error.detach())

                y_batch = self.criterion.model(self.x_batch).detach().numpy()
                plt.cla()  # 清除原有图像
                plt.plot(self.x_batch.numpy(), y_batch, label="估计y(x)")
                plt.plot(real_x, real_y, c="r", label="真实y(x)")
                plt.legend()
                plt.title("loss = {}".format(loss))
                plt.pause(0.1)
                display.clear_output(wait=True)  # 每次显示完图以后删除，达到显示动图的效果
        plt.show()

    def get_errors(self):
        return self.errors


