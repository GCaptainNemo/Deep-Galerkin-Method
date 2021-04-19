import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from matplotlib import cm


class Net(nn.Module):
    def __init__(self, hidden_layer_num, node_num):
        """
        Mapping input x, y, z, t to T
        :param hidden_layer_num:
        :param node_num:
        """
        super(Net, self).__init__()
        self.input_layer = nn.Linear(3, node_num)
        self.hidden_layers = nn.ModuleList([nn.Linear(node_num, node_num) for i in range(hidden_layer_num)])
        self.output_layer = nn.Linear(node_num, 1)

    def forward(self, x):
        o = self.activate_func(self.input_layer(x))
        for i, li in enumerate(self.hidden_layers):
            o = self.activate_func(li(o))
        out = self.output_layer(o)
        return out

    def activate_func(self, x):
        # return x * torch.sigmoid(x)
        return torch.tanh(x)