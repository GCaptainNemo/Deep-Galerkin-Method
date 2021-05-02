import torch
import torch.nn as nn
import torch.optim as optim


# dy/dx = 3, y(x) = 0 x \in [0, 4]
class SolveOde(nn.Module):
    def __init__(self, layer_num_cond, node_num_cond):
        super(SolveOde, self).__init__()
        self.node_num_cond = node_num_cond
        self.layer_num_cond = layer_num_cond
        self.input_layer_cond = nn.Linear(1, node_num_cond)
        self.hidden_layer_cond = nn.Linear(node_num_cond, node_num_cond)
        self.output_layer_cond = nn.Linear(node_num_cond, 1)

    def forward(self, x):
        x = self.activate_function(self.input_layer_cond(x))
        for i in range(self.layer_num_cond):
            s = x
            x = self.activate_function(self.hidden_layer_cond(x))
            x = x + s
        y = self.output_layer_cond(x)
        return y

    def activate_function(self, x):
        return torch.tanh(x)
