import torch
import torch.nn as nn
import torch.optim as optim


# dy/dx = 3, y(x) = 0 x \in [0, 4]
class EstimateCond(nn.Module):
    def __init__(self, depth_cond, width_cond, depth_temp, width_temp):
        super(EstimateCond, self).__init__()
        self.input_layer_cond = nn.Linear(1, width_cond)
        self.lt_cond = nn.Sequential(nn.Linear(width_cond, width_cond),
                                     nn.Tanh())
        self.hidden_layers_cond = nn.ModuleList([self.lt_cond
                                            for i in range(depth_cond)])
        self.output_layer_cond = nn.Linear(width_cond, 1)
        # ########################################################33
        self.input_layer_temp = nn.Linear(1, width_temp)
        self.lt_temp = nn.Sequential(nn.Linear(width_temp, width_temp),
                                     nn.Tanh())
        self.hidden_layers_temp = nn.ModuleList(
            [self.lt_temp for i in range(depth_temp)])
        self.output_layer_temp = nn.Linear(width_temp, 1)

    def forward(self, x):
        input_x = x
        x = self.activate_function(self.input_layer_cond(x))
        for i, lt in enumerate(self.hidden_layers_cond):
            s = x
            x = lt(x)
            x = x + s
        cond = self.output_layer_cond(x)

        # ######################################################3

        x = self.activate_function(self.input_layer_temp(input_x))
        for i, lt in enumerate(self.hidden_layers_temp):
            s = x
            x = lt(x)
            x = x + s
        temp = self.output_layer_temp(x)
        return cond, temp

    def activate_function(self, x):
        return torch.tanh(x)
