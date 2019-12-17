import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size = 5, padding = 2),
            nn.Dropout3d(0.05),
            nn.LeakyReLU(),
            nn.BatchNorm3d(4),
            nn.MaxPool3d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size = 3, padding = 1),
            nn.Dropout3d(0.05),
            nn.LeakyReLU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size = 3, padding = 1),
            nn.Dropout3d(0.05),
            nn.LeakyReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size = 3, padding = 1),
            nn.Dropout3d(0.05),
            nn.LeakyReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size = 3, padding = 1),
            nn.Dropout3d(0.05),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
        )
        self.layer6 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size = 3, padding = 1),
            nn.Dropout3d(0.05),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128),
            nn.AvgPool3d(2),
        )
        self.lastLayer = nn.Sequential(
            nn.Linear(128*4*4*4, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        x = self.lastLayer(x)
        return x
