import torch
import torch.nn as nn
from torch.nn.functional import elu

class A3CModel(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.gru = nn.GRUCell(288, 256)

        self.policy_head = nn.Linear(256, action_size)
        torch.nn.init.kaiming_normal_(self.policy_head.weight)
        self.value_head = nn.Linear(256, 1)
        torch.nn.init.kaiming_normal_(self.value_head.weight)

    def forward(self, img, hidden=None):
        """
        Returns actor, critic, hidden
        """
        x = elu(self.conv1(img))
        x = elu(self.conv2(x))
        x = elu(self.conv3(x))
        x = elu(self.conv4(x))
        x = x.view(-1, 288)
        hx = self.gru(x, hidden)
        return self.policy_head(hx), self.value_head(hx), hx
    

