#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/17 1:06
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm

te = 4
xe = 1
ye = 1
t_range = np.linspace(0, te, 100, dtype=np.float64)
x_range = np.linspace(0, xe, 100, dtype=np.float64)
y_range = np.linspace(0, ye, 100, dtype=np.float64)
_X, _Y = np.meshgrid(x_range, y_range, indexing='ij')
Z = (_X - 0.5) ** 2 + (_Y - 0.5) ** 2
Z_surface = np.reshape(Z, (x_range.shape[0], y_range.shape[0]))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_zlim([-1, 1])
ax.plot_surface(_X, _Y, Z_surface, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('k(m^2/s) ')
plt.show()
