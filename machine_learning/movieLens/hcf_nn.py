
import torch
import torch.nn.functional as F
import torch.nn as nn


class Hcf(nn.Module):
    # input is concat(u+, w, u-)
    def __init__(self, num_feature=64):
        super(Hcf, self).__init__()
        self.in_feature = 512
        self.conv1 = nn.Conv2d(2, num_feature, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(1, num_feature, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_feature, num_feature * 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_feature, num_feature * 2, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.in_feature, 1)

        self.b1 = nn.BatchNorm2d(num_feature * 2)
        self.b2 = nn.BatchNorm2d(num_feature * 2)

    def forward(self, u, w):
        out1 = F.relu(self.b1(self.conv1(u)))  # u is concat(pos_u, neg_u)
        out2 = F.relu(self.b2(self.conv2(w)))
        out1 = F.relu(self.b3(self.conv3(out1)))
        out2 = F.relu(self.b4(self.conv4(out2)))
        out = torch.cat((out1, out2), 0)
        out = self.fc1(out)

        return out
