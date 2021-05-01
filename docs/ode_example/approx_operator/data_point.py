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
    def __init__(self, cheby_degree):
        """
        :param batch_num: y num
        :param cheby_degree: cheybshev polynomials degree
        """
        self.cheby_degree = cheby_degree
        self.proportion_bound = 5

    def sample_chebyshev_batch(self):
        """ sample from [0, 1] Chebyshev(Orthogonol) polynomials """
        coefficient = (torch.rand([1, self.cheby_degree]) - 0.5) * 2 * \
                      self.proportion_bound
        origin_x = torch.rand([100, 1])
        ux = self.cal_g_u_y_chebyshev(origin_x, coefficient)

        # y sample uniformly on [0, 1]
        trunk_y = torch.rand([1, 1])
        guy = self.cal_g_u_y_chebyshev(trunk_y, coefficient)
        x_ux_y_guy = torch.cat([origin_x.t(), ux.t(), trunk_y, guy], dim=1)
        return origin_x, x_ux_y_guy

    def cal_g_u_y_chebyshev(self, y, coefficient):
        """ calculate G(u)(y) """
        batch_num = y.shape[0]
        linshi_matrix = torch.zeros([batch_num, self.cheby_degree])
        linshi_matrix[:, 0] = 1
        linshi_matrix[:, 1] = (y.reshape([-1]) - 0.5) * 2
        for i in range(2, self.cheby_degree):
            linshi_matrix[:, i] = 2 * linshi_matrix[:, 1] * linshi_matrix[:, i - 1] - \
                                  linshi_matrix[:, i - 2]
        return linshi_matrix @ coefficient.t()

    def create_dataset(self, function_num):
        x_ux_y_guy = self.sample_chebyshev_batch()
        for i in range(1, function_num):
            x_ux_y_guy = torch.cat([x_ux_y_guy, self.sample_chebyshev_batch()], dim=0)
        with open("chebyshev_ux.pkl", "wb") as f:
            pickle.dump(x_ux_y_guy, f)


class CreateGpData:
    def __init__(self, l):
        """
        [0, 1]
        create gaussian process data as function space {f(x)}
        :param l: radial basis function(length), larger l leads to smoother f(x)
        """
        self.l = l

    def caculate_covariance_matrix(self):
        x_origin = torch.rand([100, 1])
        covariance = x_origin ** 2 + (x_origin ** 2).t() - 2 * x_origin @ x_origin.t()
        covariance = torch.exp(-covariance / self.l)
        covariance = covariance + torch.eye(100) * 1e-5
        result = torch.cholesky(covariance, upper=False)
        return x_origin, result

    def sample_from_gp(self):
        origin_x, left_tri_matrix = self.caculate_covariance_matrix()
        u_x = (left_tri_matrix @ origin_x).reshape([1, -1])
        origin_x = origin_x.reshape([1, -1])
        y = torch.rand([1, 1])
        # One-dimensional linear interpolation.
        linshi_x, index_x = origin_x.sort(dim=1)
        linshi_x = linshi_x.squeeze(dim=0).numpy()
        linshi_u_x = u_x[0, index_x].squeeze(dim=0).numpy()
        # linshi = torch.cat([origin_x, u_x], dim=0).sort(dim=1)
        guy = torch.from_numpy(np.interp(y, linshi_x, linshi_u_x)).to(dtype=torch.float32)
        data = torch.cat([origin_x, u_x, y, guy], dim=1)
        return data

    def create_dataset(self, function_num):
        data = self.sample_from_gp()
        for i in range(1, function_num):
            data = torch.cat([data, self.sample_from_gp()], dim=0)
        with open("gaussian_ux.pkl", "wb") as f:
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
        x = self.trainset[num, 0:100].numpy()
        u = self.trainset[num, 100:200].numpy()

        # x = np.linspace(0, 4, 100)
        plt.figure(1)
        # plt.subplot(121)
        plt.scatter(x, u, c="r", s=2)
        # plt.subplot(122)
        plt.scatter(self.trainset[num, 200], self.trainset[num, 201], s=2)
        plt.show()


if __name__ == "__main__":
    # create_gp_obj = CreateGpData(0.2)
    # create_gp_obj.create_dataset(1000)

    # create_data_obj = CreateChebyData(10)
    # create_data_obj.create_dataset(function_num=1000)

    # dir = "chebyshev_ux.pkl"
    dir = "gaussian_ux.pkl"
    test_obj = TestDataset(dir)
    test_obj.plot(59)


