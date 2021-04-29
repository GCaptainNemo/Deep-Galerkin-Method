#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/25 22:58

import torch.optim as optim
import matplotlib.pyplot as plt
import torch


class Train:
    def __init__(self, dataloader, criterion):
        self.criterion = criterion
        self.errors = []
        self.x_batch = torch.linspace(0, 4, 400,
                                      dtype=torch.float32).reshape(-1, 1)
        self.dataloader = dataloader

    def train(self, epoch, lr):
        optimizer = optim.Adam(self.criterion.model.parameters(), lr)
        # optimizer_trunk = optim.Adam(self.criterion.trunk_net.parameters(), lr)

        for e in range(epoch):
            avg_loss = 0
            for i, data in enumerate(self.dataloader):
                # data.cuda()
                # print("data.device = ", data.device)

                optimizer.zero_grad()
                loss = self.criterion.loss_func(data)
                avg_loss = avg_loss + float(loss.item())
                loss.backward()
                optimizer.step()
            if e % 20 == 19:
                loss = avg_loss / 20
                print("Epoch {} - lr {} -  loss: {}".format(e, lr, loss))
                avg_loss = 0
                self.errors.append(loss)
                # 清除原有图像

    def get_errors(self):
        return self.errors


