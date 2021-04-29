import torch.nn as nn
import torch


class OperatorPointApprox(nn.Module):
    def __init__(self, trunk_depth, trunk_width):
        super(OperatorPointApprox, self).__init__()
        self.conv1 = nn.Conv1d(2, 8, 1)
        self.conv2 = nn.Conv1d(8, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)

        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 10, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(10)

        # ##############################################3
        self.trunk_input = nn.Linear(1, trunk_width)
        self.trunk_hidden = nn.Linear(trunk_width, trunk_width)
        self.trunk_output = nn.Linear(trunk_width, 10)
        self.trunk_depth = trunk_depth

        # ############################################
        self.bias = nn.Parameter(torch.tensor([0], dtype=torch.float32,
                                              requires_grad=True))
        self.register_parameter("bias", self.bias)

    def branch_net(self, x):
        x = x.reshape([-1, 2, 100])
        # x = torch.unsqueeze(x, dim=1)
        x = self.activate(self.bn1(self.conv1(x)))
        x = self.activate(self.bn2(self.conv2(x)))
        x = self.activate(self.bn3(self.conv3(x)))
        x = self.activate(self.bn4(self.conv4(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        # print(x.shape)
        x = x.view(-1, 1024) # N x 1024

        x = self.activate(self.bn5(self.fc5(x)))
        x = self.activate(self.bn6(self.fc6(x)))
        x = self.bn7(self.fc7(x))
        return x

    def trunk_net(self, x):
        x = self.activate(self.trunk_input(x))
        for i in range(self.trunk_depth):
            x = self.trunk_hidden(x)
            input_ = x
            x = self.activate(x)
            x + input_
        x = self.activate(self.trunk_output(x))
        return x

    def forward(self, x):
        branch_input = x[:, :200]
        trunk_input = x[:, 200].reshape(-1, 1)
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        guy = branch_output @ trunk_output.t() + self.bias
        return guy

    def activate(self, x):
        # return torch.tanh(x)
        return torch.relu(x)
