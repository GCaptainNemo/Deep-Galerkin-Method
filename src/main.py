#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/4/14 23:32 
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from libs import *
from src.train import *
from src.model import *
from src.pde_model import *

net = Net(2, 20)
te = 4
xe = 1
ye = 1
heatequation = Heat(net, te, xe, ye)

train = Train(net, heatequation, BATCH_SIZE=2**8)

train.train(epoch=105, lr=0.0001)

torch.save(net, '../model/net_model.pkl')

errors = train.get_errors()

#plot errors
fig = plt.figure()
plt.plot(np.log(errors), '-b', label='Errors')
plt.title('Training Loss', fontsize=10)
path = "../pictures/trainingloss.png"
# plt.show()
plt.savefig(path)
plt.close(fig)

# net = torch.load('net_model.pkl')

t_range = np.linspace(0, te, 100, dtype=np.float64)
x_range = np.linspace(0, xe, 100, dtype=np.float64)
y_range = np.linspace(0, ye, 100, dtype=np.float64)

data = np.empty((3, 1))

k = 0
for _t in t_range:
    if _t % 100 == 0:
        TrueZ = []
        Z = []
        data[0] = _t
        for _x in x_range:
            data[1] = _x
            for _y in y_range:
                data[2] = _y
                indata = torch.Tensor(data.reshape(1, -1))
                Zdata = net(indata).detach().cpu().numpy()
                Z.append(Zdata)
                TrueZ.append(np.sin(np.pi*_t)*np.sin(np.pi*_x)*np.sin(np.pi*_y))


        _X, _Y = np.meshgrid(x_range, y_range, indexing='ij')

        Z_surface = np.reshape(Z, (x_range.shape[0], y_range.shape[0]))
        True_Z_surface = np.reshape(TrueZ, (x_range.shape[0], y_range.shape[0]))

        # plot the approximated values
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlim([-1, 1])
        ax.plot_surface(_X, _Y, Z_surface, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        path = "../pictures/approxmate_solution/%i.png" % k
        plt.savefig(path)
        plt.close(fig)
        # plot the exact solution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlim([-1, 1])
        ax.plot_surface(_X, _Y, True_Z_surface, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        path = "../pictures/exact_solution/%i.png" % k
        plt.savefig(path)
        plt.close(fig)
        k = k + 1

