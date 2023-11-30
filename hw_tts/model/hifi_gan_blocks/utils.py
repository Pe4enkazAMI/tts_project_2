import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

class SubBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation) -> None:
        super().__init__()
        layers = []

        for _ in range(len(dilation)):
            block = [
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=kernel_size,
                                    dilation=dilation[_],
                                    padding=int(dilation[_] * (kernel_size - 1) / 2))),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                padding=int((kernel_size - 1) / 2)))
            ]
            layers += block
        self.block = nn.ModuleList(layers)

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.block):
            if len(layer.state_dict()) != 0:
                try:
                    remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()
        remove_weight_norm(self.shortcut)

    def forward(self, x):
        res = 0
        for layer in self.block:
            res = res + layer(x)
        return res
    
class MRF(nn.Module):
    def __init__(self, channels, resblock_kernels, resblock_dilations) -> None:
        super().__init__()
        resblocks = []
        for i in range(len(resblock_kernels)):
            resblocks += [SubBlock(channels, resblock_kernels[i], resblock_dilations[i])]

        self.resblocks = nn.ModuleList(resblocks)

    def forward(self, x):
        res = 0
        for layer in self.resblocks:
            res = res + layer(x)
        return res

