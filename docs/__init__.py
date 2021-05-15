# import torch.nn as nn
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # self.conv1 = nn.Conv2d(1, 1, 3)
#         self.layers = nn.ModuleList([nn.Conv2d(1, 1, 3) for i in range(3)])
#
#     def forward(self, x):
#         # for i in range(3):
#         #     x = self.conv1(x)
#         for i, li in enumerate(self.layers):
#             x = li(x)
#         return x
#
# model = Net()
#
# print(model)
# print()
#
# for name, parameters in model.named_parameters():
#     print(name, ':', parameters.size())