import torch
import torch.nn as nn
import torch.optim as optim


# dy/dx = 3, y(x) = 0 x \in [0, 4]
class EstimateCond(nn.Module):
    def __init__(self, layer_num_cond, node_num_cond, layer_num_temp, node_num_temp):
        super(EstimateCond, self).__init__()
        self.node_num_cond = node_num_cond
        self.layer_num_cond = layer_num_cond
        self.input_layer_cond = nn.Linear(1, node_num_cond)
        self.hidden_layer_cond = nn.Linear(node_num_cond, node_num_cond)
        self.output_layer_cond = nn.Linear(node_num_cond, 1)
        # ########################################################33
        self.node_num_temp = node_num_temp
        self.layer_num_temp = layer_num_temp
        self.input_layer_temp = nn.Linear(1, node_num_temp)
        self.hidden_layer_temp = nn.Linear(node_num_temp, node_num_temp)
        self.output_layer_temp = nn.Linear(node_num_temp, 1)

    def forward(self, x):
        input_x = x
        x = self.activate_function(self.input_layer_cond(x))
        for i in range(self.layer_num_cond):
            s = x
            x = self.activate_function(self.hidden_layer_cond(x))
            x = x + s
        cond = self.output_layer_cond(x)

        # ######################################################3

        x = self.activate_function(self.input_layer_temp(input_x))
        for i in range(self.layer_num_temp):
            s = x
            x = self.activate_function(self.hidden_layer_temp(x))
            x = x + s
        temp = self.output_layer_temp(x)
        return cond, temp

    def activate_function(self, x):
        return torch.tanh(x)
