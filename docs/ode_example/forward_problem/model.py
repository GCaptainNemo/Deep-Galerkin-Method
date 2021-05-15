import torch
import torch.nn as nn
import torch.optim as optim


# dy/dx = 3, y(x) = 0 x \in [0, 4]
class SolveOde(nn.Module):
    def __init__(self, depth, width):
        super(SolveOde, self).__init__()
        self.input_layer_cond = nn.Linear(1, width)
        self.lbr = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh()
        )
        self.hidden_layers = nn.ModuleList([self.lbr for i in range(depth)])
        self.output_layer_cond = nn.Linear(width, 1)

    def forward(self, x):
        x = self.activate_function(self.input_layer_cond(x))
        for i, lbt in enumerate(self.hidden_layers):
            s = x
            x = lbt(x)
            x = x + s
        y = self.output_layer_cond(x)
        return y

    def activate_function(self, x):
        return torch.tanh(x)
