#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/14 23:33 

import torch
from torch.autograd import Variable
import numpy as np


class Heat:
    def __init__(self, net_model, te, xe, ye):
        """
        :param net_model: mappint x, y, t to T
        :param te: time step
        :param xe: x step
        :param ye: y step
        """
        # self.cond_function = lambda x, y: 5 * (x - 0.5) ** 2 + 5 * (y - 0.5) ** 2 + 5
        self.cond_function = lambda x, y: 1
        self.net = net_model
        self.te = te
        self.xe = xe
        self.ye = ye

    def sample(self, size):
        te = self.te
        xe = self.xe
        ye = self.ye
        # t_x_y = [ti, xi, yi]i=1...,size (size x 3 tensor)
        t_x_y = torch.cat((torch.rand([size, 1]) * te, torch.rand([size, 1]) * xe, torch.rand([size, 1]) * ye), dim=1)
        x_y_initial = torch.cat((torch.zeros(size, 1), torch.rand([size, 1]) * xe, torch.rand([size, 1]) * ye), dim=1)
        # boundary condition [0, 1] x [0, 1] = 0
        boundary_left = torch.cat((torch.rand([size, 1]) * te, torch.zeros([size, 1]), torch.rand(size, 1) * ye),
                                    dim=1)
        boundary_right = torch.cat((torch.rand([size, 1]) * te, torch.ones([size, 1]) * xe, torch.rand(size, 1) * ye),
                                     dim=1)
        boundary_up = torch.cat((torch.rand([size, 1]) * te, torch.rand([size, 1]) * xe, torch.ones(size, 1) * ye),
                                  dim=1)
        boundary_down = torch.cat((torch.rand([size, 1]) * te, torch.rand([size, 1]) * xe, torch.zeros(size, 1)),
                                    dim=1)
        return t_x_y, x_y_initial, boundary_left, boundary_right, boundary_up, boundary_down

    def loss_func(self, size):
        t_x_y, x_initial, x_boundary_left, x_boundary_right, x_boundary_up, x_boundary_down = self.sample(size=size)
        t_x_y = Variable(t_x_y, requires_grad=True)

        jacob_matrix = torch.autograd.grad(self.net(t_x_y), t_x_y, grad_outputs=torch.ones_like(self.net(t_x_y)), create_graph=True)
        dtemp_dt = jacob_matrix[0][:, 0].reshape(-1, 1)  # transform the vector into a column vector
        dtemp_dx = jacob_matrix[0][:, 1].reshape(-1, 1)
        dtemp_dy = jacob_matrix[0][:, 2].reshape(-1, 1)
        # du/dxdx
        dtemp_dxx = torch.autograd.grad(dtemp_dx, t_x_y, grad_outputs=torch.ones_like(dtemp_dx), create_graph=True)[0][:, 1].reshape(-1, 1)
        # du/dydy
        dtemp_dyy = torch.autograd.grad(dtemp_dy, t_x_y, grad_outputs=torch.ones_like(dtemp_dy), create_graph=True)[0][:, 2].reshape(-1, 1)

        # setting conduct heat parameter
        # conduct_heat_par = 5 * (t_x_y[:, 1] - 0.5) ** 2 + 5 * (t_x_y[:, 2] - 0.5) ** 2 + 5
        conduct_heat_par = self.cond_function(t_x_y[:, 1], t_x_y[:, 2])
        heat_source = np.pi * (torch.cos(np.pi * t_x_y[:, 0])) * (torch.sin(np.pi * t_x_y[:, 1])) * (torch.sin(np.pi * t_x_y[:, 2])) \
            + 2 * np.pi * np.pi * (torch.sin(np.pi * t_x_y[:, 0])) * (torch.sin(np.pi * t_x_y[:, 1])) * (
                torch.sin(np.pi * t_x_y[:, 2]))
        # add conduct heat coefficient
        diff_error = (dtemp_dt - conduct_heat_par * (dtemp_dxx + dtemp_dyy)
                      - heat_source.reshape(-1, 1)) ** 2

        # initial condition(T_init = 0)
        init_error = (self.net(x_initial)) ** 2
        # boundary condition(T_boundary = 0)
        bd_left_error = (self.net(x_boundary_left)) ** 2
        bd_right_error = (self.net(x_boundary_right)) ** 2
        bd_up_error = (self.net(x_boundary_up)) ** 2
        bd_down_error = (self.net(x_boundary_down)) ** 2
        error = torch.mean(diff_error + init_error + bd_left_error + bd_right_error + bd_up_error + bd_down_error)
        return error
