#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/28 14:29 
import torch
import pickle
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt


class CreateChebyData:
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


class CreateGpData:
    def __init__(self, l, batch_size):
        """
        create gaussian process data as function space {f(x)}
        :param l: radial basis function(length), larger l leads to smoother f(x)
        """
        self.l = l
        self.left_tri_matrix = self.caculate_covariance_matrix()
        self.batch_size = batch_size

    def caculate_covariance_matrix(self):
        x = torch.linspace(0, 4, 100)
        covariance = torch.zeros([100, 100], dtype=torch.float32)
        for i in range(100):
            covariance[i, :] = np.exp(-((x[i] - x) ** 2) / self.l)
        e, v = torch.eig(covariance, eigenvectors=False)
        min_eigval = torch.min(e)
        print(min_eigval)
        if min_eigval < 0:
            covariance = covariance + torch.eye(100) * 1e-5
        result = torch.cholesky(covariance, upper=False)
        return result

    def sample_from_gp(self):
        print(self.left_tri_matrix.shape)
        u = torch.ones([self.batch_size, 1]) @ \
            (self.left_tri_matrix @ \
            torch.randn([100, 1])).reshape([1, -1])
        y = torch.rand([self.batch_size, 1]) * 4
        guy = torch.zeros(y.shape)
        x_step = 4 / 99
        for i in range(self.batch_size):
            index = int(y[i, 0].item() // x_step)
            # print(index)
            u_left = u[0, index]
            u_right = u[0, index + 1]
            fraction = (y[i, 0] - index * x_step) / x_step
            guy[i, 0] = fraction * u_right + (1 - fraction) * u_left
        data = torch.cat([u, y, guy], dim=1)
        return data

    def create_dataset(self, function_num):
        data = self.sample_from_gp()
        for i in range(1, function_num):
            data = torch.cat([data, self.sample_from_gp()], dim=0)
        with open("gaussian.pkl", "wb") as f:
            pickle.dump(data, f)


class CustomDataset(data.Dataset):
    def __init__(self, dir):
        self.open_dataset(dir)

    def open_dataset(self, dir):
        with open(dir, "rb") as f:
            self.trainset = pickle.load(f)
        print(self.trainset.shape)

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
        plt.scatter(x, u, c="r", s=2)
        # plt.subplot(122)
        plt.scatter(self.trainset[num, 100], self.trainset[num, 101], s=2)
        plt.show()


if __name__ == "__main__":
    # create_gp_obj = CreateGpData(0.2, 10)
    # create_gp_obj.create_dataset(10)
    # create_gp_obj.caculate_covariance_matrix()
    # create_data_obj = CreateDataset(100, 10)
    # create_data_obj.create_dataset(function_num=100)
    dir = "chebyshev.pkl"
    # dir = "gaussian.pkl"
    test_obj = TestDataset(dir)
    test_obj.plot(1009)