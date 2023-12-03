import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch import nn

def padding_size(kernel, stride, dillation):
    return dillation * (kernel - 1) // 2

class MRFBlock(nn.Module):
    def __init__(self, channels, kernel,  dilations):
        super().__init__()

        self.kernel = kernel
        self.dilations = dilations

        layers = []

        for m in range(len(dilations)):
            layer = nn.Sequential(
                nn.LeakyReLU(),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel, 
                          dilation=dilations[m], 
                          padding=padding_size(kernel, 1, dilations[m]))),
                nn.LeakyReLU(),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel, 
                          dilation=1, 
                          padding=padding_size(kernel, 1, 1))),
            )
            layers.append(layer)
        
        self.block = nn.ModuleList(layers)

    def forward(self, x):
        result = 0
        for layer in self.block:
            result = result + layer(x)
        return result


class MRF(nn.Module):
    def __init__(self, channels, resblock_kernels, resblock_dilations):
        super().__init__()

        resblocks = []
        for i in range(len(resblock_kernels)):
            resblocks.append(MRFBlock(channels, resblock_kernels[i], resblock_dilations[i]))

        self.resblocks = nn.ModuleList(resblocks)

    def forward(self, x):
        result = 0
        for block in self.resblocks:
            result = result + block(x)
        return result

class Generator(nn.Module):
    def __init__(self, input_channels, hidden_channels, upsample_kernels,
                 upsample_stride, resblock_kernels,
                 resblock_dilations) -> None:
        super().__init__()

        self.in_proj = nn.Conv1d(input_channels, hidden_channels, 7, 
                                  padding=3)

        blocks = []
        current_channels = hidden_channels
        for i in range(len(upsample_kernels)):
            upsample_block = nn.ConvTranspose1d(current_channels,
                                          current_channels // 2,
                                          upsample_kernels[i],
                                          upsample_stride[i],
                                          padding=(upsample_kernels[i] - upsample_stride[i]) // 2)
            mrf_block = MRF(current_channels // 2,
                      resblock_kernels,
                      resblock_dilations)
            block = nn.Sequential(upsample_block, mrf_block)
            blocks.append(block)

            current_channels = current_channels // 2

        self.blocks = nn.Sequential(*blocks)

        self.out_proj = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(current_channels, 1, 7, 
                      padding=3),
            nn.Tanh()
        )

    def forward(self, mel):
        mel = self.in_proj(mel)
        mel = self.blocks(mel)
        audio = self.out_proj(mel)
        return {"pred_audio": audio}