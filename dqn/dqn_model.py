import torch
import torch.nn as nn
import torch.nn.functional as F

ENV_DIM = 42
    
class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(288, 64)
        self.head = nn.Linear(64, action_size)

    def forward(self, x):
        # x = x.view(-1, 1, ENV_DIM, ENV_DIM)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.head(x)