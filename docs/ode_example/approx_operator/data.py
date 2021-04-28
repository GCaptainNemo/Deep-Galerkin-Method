#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/28 14:29 
import torch
import pickle
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt


class CreateDataset:
    def __init__(self, batch_num, cheby_degree):
        """

        :param batch_num: y num
        :param cheby_degree: cheybshev polynomials degree
        """
        self.batch_num = batch_num
        self.cheby_degree = cheby_degree
        # Cheybshev polynomials(first two), [-1, 1]
        self.basis_functions = self.create_chebyshev_basis()
        self.proportion_bound = 5

    def create_chebyshev_basis(self):
        basis_functions = torch.zeros([self.cheby_degree, 100])
        basis_functions[0, :] = 1
        x1 = (torch.linspace(0, 4, 100) - 2) / 2
        basis_functions[1, :] = x1
        for i in range(2, self.cheby_degree):
            basis_functions[i, :] = 2 * basis_functions[1, :] * basis_functions[i - 1, :] \
                                    - basis_functions[i - 2, :]
        return basis_functions

    def sample_chebyshev_batch(self):
        """ sample from [0, 4] Chebyshev(Orthogonol) polynomials """
        coefficient = (torch.rand([1, self.cheby_degree]) - 0.5) * 2 * \
                      self.proportion_bound
        branch_u = torch.ones([self.batch_num, 1]) @ coefficient @ \
                   self.basis_functions

        # y sample uniformly on [0, 4]
        trunk_y = torch.rand([self.batch_num, 1]) * 4
        guy = self.cal_g_u_y_chebyshev(trunk_y, coefficient)
        u_y_guy = torch.cat([branch_u, trunk_y, guy], dim=1)
        return u_y_guy

    def cal_g_u_y_chebyshev(self, y, coefficient):
        """ calculate G(u)(y) """
        batch_num = y.shape[0]
        linshi_matrix = torch.zeros([batch_num, self.cheby_degree])
        linshi_matrix[:, 0] = 1
        linshi_matrix[:, 1] = (y.reshape([-1]) - 2) / 2
        for i in range(2, self.cheby_degree):
            linshi_matrix[:, i] = 2 * linshi_matrix[:, 1] * linshi_matrix[:, i - 1] - \
                                  linshi_matrix[:, i - 2]
        return linshi_matrix @ coefficient.t()

    def create_dataset(self, function_num):
        u_y_guy = self.sample_chebyshev_batch()
        for i in range(1, function_num):
            u_y_guy = torch.cat([u_y_guy, self.sample_chebyshev_batch()], dim=0)
        with open("chebyshev.pkl", "wb") as f:
            pickle.dump(u_y_guy, f)


class CustomDataset(data.Dataset):
    def __init__(self, dir):
        self.open_dataset(dir)

    def open_dataset(self, dir):
        with open(dir, "rb") as f:
            self.trainset = pickle.load(f)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.trainset = self.trainset.to(device)
        # print("trainset_device = ", self.trainset.device)

    def __len__(self):
        return self.trainset.shape[0]

    def __getitem__(self, item):
        return self.trainset[item, :]


class TestDataset:
    def __init__(self, dir):
        with open(dir, "rb") as f:
            self.trainset = pickle.load(f)

        print(self.trainset.shape)

    def plot(self, num):
        u = self.trainset[num, :100].numpy()
        x = np.linspace(0, 4, 100)
        plt.figure(1)
        # plt.subplot(121)
        plt.scatter(x, u, c="r")
        # plt.subplot(122)
        plt.scatter(self.trainset[num, 100], self.trainset[num, 101])
        plt.show()


if __name__ == "__main__":
    # create_data_obj = CreateDataset(100, 10)
    # create_data_obj.create_dataset(function_num=100)
    dir = "chebyshev.pkl"
    test_obj = TestDataset(dir)
    test_obj.plot(1100)