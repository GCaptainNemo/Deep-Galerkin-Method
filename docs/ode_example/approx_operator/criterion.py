#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/28 14:29 

import torch


class Criterion:
    def __init__(self, model):
        self.model = model

    def loss_func(self, data):
        # # dim = [batch_num, 10]
        input_data = data[:, :101]
        guy = data[:, -1].reshape([-1, 1])
        estimate_guy = self.model(input_data)
        mse = torch.mean((estimate_guy - guy) ** 2)
        return mse

