import torch.nn as nn
import torch

class Flatten(nn.Module):
    """Custom flatten layer for reshaping tensors."""
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ResidualBlock(nn.Module):
    """Residual Block with skip connections."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path (Convolution layers)
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut connection (Identity mapping or 1x1 Conv)
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.main_branch(x)
        x += residual  
        return self.activation(x)


class ResNet(nn.Module):
    """ResNet Model for Classification."""
    def __init__(self):
        super(ResNet, self).__init__()

        # Initial Convolutional Layer
        self.initial_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual Blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            nn.Dropout(p=0.5),  
            ResidualBlock(256, 512, stride=2)
        )

        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.res_blocks(x)
        x = self.classifier(x)
        return x
