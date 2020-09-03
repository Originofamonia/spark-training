
import torch
import torch.nn.functional as F
import torch.nn as nn


class Hcf(nn.Module):
    # input is (u+, u-, v+, v-) all are vectors, output is P(i, j)
    def __init__(self, in_feature=32, hidden_feature=128):
        super(Hcf, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        # self.conv1 = nn.Conv2d(2, num_feature, kernel_size=3, stride=1, padding=2)
        # self.conv2 = nn.Conv2d(1, num_feature, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(num_feature, num_feature * 2, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(num_feature, num_feature * 2, kernel_size=3, stride=1)
        self.fc_u1 = nn.Linear(self.in_feature, self.hidden_feature)
        self.fc_u2 = nn.Linear(self.hidden_feature, self.hidden_feature)
        self.fc_v1 = nn.Linear(self.in_feature, self.hidden_feature)
        self.fc_v2 = nn.Linear(self.hidden_feature, self.hidden_feature)
        self.fc3 = nn.Linear(self.hidden_feature, 2)

        self.b1 = nn.BatchNorm1d(self.hidden_feature)
        self.b2 = nn.BatchNorm1d(self.hidden_feature)

    def forward(self, u, v):
        u1 = F.relu(self.b1(self.fc_u1(u)))
        u1 = F.relu(self.b1(self.fc_u2(u1)))
        v1 = F.relu(self.b2(self.fc_v1(v)))
        v1 = F.relu(self.b2(self.fc_v2(v1)))
        out = u1 + v1
        out = self.fc3(out)
        return out
