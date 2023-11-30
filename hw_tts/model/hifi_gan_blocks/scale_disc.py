import torch.nn as nn
from torch.nn.utils import weight_norm


class ScaleDiscriminator(nn.Module):
    def __init__(self, norm=False):
        super().__init__()
        convs = [
            nn.Conv1d(1, 16, 15, 1, padding=7),
            nn.Conv1d(16, 64, 41, 4, groups=4, padding=20),
            nn.Conv1d(64, 256, 41, 4, groups=16, padding=20),
            nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20),
            nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ]

        out_conv = nn.Conv1d(1024, 1, 3, 1, padding=1)

        if norm:
            convs = [weight_norm(module) for module in convs]
            out_conv = weight_norm(out_conv)
        
        self.convs = nn.ModuleList(convs)
        self.out_conv = out_conv

        self.activation = nn.LeakyReLU()

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = self.activation(conv(x))
            features.append(x)

        x = self.out_conv(x)
        features.append(x)
        x = x.view(x.shape[0], -1)
        
        return x, features