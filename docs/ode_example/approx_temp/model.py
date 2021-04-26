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

    def forward(self, x):
        input_x = x
        x = self.activate_function(self.input_layer_temp(input_x))
        for i in range(self.layer_num_temp):
            s = x
            x = self.activate_function(self.hidden_layer_temp(x))
            x = x + s
        temp = self.output_layer_temp(x)
        return temp

    def activate_function(self, x):
        return torch.tanh(x)
