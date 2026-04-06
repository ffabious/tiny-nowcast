from torch import nn

class TinyNowcastModel(nn.Module):
    def __init__(self, in_channels = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
    
class BaselineModel(nn.Module):
    # Return last input frame as prediction
    def __init__(self, in_channels = 4):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        return x[:, -1:, :, :]