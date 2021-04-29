import torch
import matplotlib.pyplot as plt
import numpy as np
# cheby_degree = 5

#
# def create_basis_functions():
#     basis_functions = torch.zeros([cheby_degree, 100])
#     basis_functions[0, :] = torch.ones([100])
#     x1 = (torch.linspace(0, 4, 100) - 2) / 2
#     basis_functions[1, :] = x1
#     for i in range(2, cheby_degree):
#         basis_functions[i, :] = 2 * basis_functions[1, :] * basis_functions[i - 1, :] \
#                                 - basis_functions[i - 2, :]
#     return basis_functions
#
#
# basis_functions = create_basis_functions()
# x = torch.linspace(0, 4, 100)
# plt.figure(1)
# plt.plot(x, basis_functions[2, :])
# plt.show()


def caculate_covariance_matrix():
    N = 100
    x = np.linspace(0, 4, N)
    covariance = np.zeros([N, N], dtype=np.float)
    for i in range(N):
        # for j in range(100):
        covariance[i, :] = np.exp(-((x[i] - x) ** 2) / 0.2)
    # s, u = np.linalg.eig(covariance)
    # print(s)
    print(np.min(np.linalg.eigvals(covariance)).real)
    # print(np.all(np.linalg.eigvals(covariance) >= -0.00000000000001))
    # print(s.real > 0)

caculate_covariance_matrix()
