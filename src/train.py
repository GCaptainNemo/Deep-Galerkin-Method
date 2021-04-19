#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/14 23:31
import torch.optim as optim


class Train():
    def __init__(self, net, heateq, batch_size):
        self.errors = []
        self.BATCH_SIZE = batch_size
        self.net = net
        self.model = heateq

    def train(self, epoch, lr):
        optimizer = optim.Adam(self.net.parameters(), lr)
        avg_loss = 0
        for e in range(epoch):
            optimizer.zero_grad()
            loss = self.model.loss_func(self.BATCH_SIZE)
            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            if e % 50 == 49:
                loss = avg_loss/50
                print("Epoch {} - lr {} -  loss: {}".format(e, lr, loss))
                avg_loss = 0

                error = self.model.loss_func(2**8)
                self.errors.append(error.detach())

    def get_errors(self):
        return self.errors