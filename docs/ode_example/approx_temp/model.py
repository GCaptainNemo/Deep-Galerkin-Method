import torch
import torch.nn as nn
import torch.optim as optim


# dy/dx = 3, y(x) = 0 x \in [0, 4]
class ApproxTemp(nn.Module):
    def __init__(self, layer_num_temp, node_num_temp):
        super(ApproxTemp, self).__init__()
        self.node_num_temp = node_num_temp
        self.layer_num_temp = layer_num_temp
        self.input_layer_temp = nn.Linear(1, node_num_temp)
        self.hidden_layer_temp = nn.Linear(node_num_temp, node_num_temp)
        self.output_layer_temp = nn.Linear(node_num_temp, 1)
        # self.bias = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        # self.register_parameter("bias", self.bias)

        # self.bn_layer = nn.BatchNorm1d(num_features=node_num_temp, affine=True)

    def forward(self, x):
        input_x = x
        x = self.activate_function(self.input_layer_temp(input_x))
        for i in range(self.layer_num_temp):
            s = x
            self.hidden_layer_temp(x)
            # x = self.bn_layer(x)
            x = self.activate_function(x)
            x = x + s
        # x = self.activate_function(x)
        temp = self.output_layer_temp(x)
        return temp

    def activate_function(self, x):
        # return torch.relu(x)
        return torch.tanh(x)
#
#
# class Bias(nn.Module):
#     def __init__(self):
#         super(Bias, self).__init__()
#
#
#     def forward(self, x):
#         print('bias = ', self.bias.data)
#         return x + self.bias

