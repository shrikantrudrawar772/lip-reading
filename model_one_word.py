import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()

        # 3D CNN
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool3d((1, 2, 2))

        # 🔥 GLOBAL AVERAGE POOL (VERY IMPORTANT)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Fully connected
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # x: (B, C, T, H, W)

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # 🔥 Reduce huge tensor to (B, 128, 1, 1, 1)
        x = self.global_pool(x)

        x = x.view(x.size(0), -1)  # (B, 128)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
