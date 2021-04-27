import torch
import torch.nn as nn
import torch.optim as optim


# dy/dx = 3, y(x) = 0 x \in [0, 4]
class ApproxSolve(nn.Module):
    def __init__(self, layer_num, node_num):
        super(ApproxSolve, self).__init__()
        self.node_num = node_num
        self.layer_num = layer_num
        self.input_layer = nn.Linear(1, node_num)
        self.hidden_layer = nn.Linear(node_num, node_num)
        self.output_layer = nn.Linear(node_num, 1)

    def forward(self, x):
        x = self.activate_function(self.input_layer(x))
        for i in range(self.layer_num):
            s = x
            x = self.activate_function(self.hidden_layer(x))
            x = x + s
        y = self.output_layer(x)
        return y

    def activate_function(self, x):
        return torch.tanh(x)

        # return torch.relu(x)
