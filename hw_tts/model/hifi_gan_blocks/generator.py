import torch
import torch.nn as nn 
import torch.nn.functional as F
from .utils import MRF


class Generator(nn.Module):
    def __init__(self, input_channels, hidden_channels, upsample_kernels,
                 upsample_stride, resblock_kernels,
                 resblock_dilations) -> None:
        super().__init__()

        self.in_proj = nn.Conv1d(input_channels, hidden_channels, 7, 
                                  padding= (7 - 1) // 2  )

        blocks = []
        current_channels = hidden_channels
        for i in range(len(upsample_kernels)):
            upsample_block = nn.ConvTranspose1d(current_channels,
                                          current_channels >> 1,
                                          upsample_kernels[i],
                                          upsample_stride[i],
                                          padding=(upsample_kernels[i] - upsample_stride[i]) >> 1)
            mrf_block = MRF(current_channels >> 1,
                      resblock_kernels,
                      resblock_dilations)
            block = nn.Sequential(upsample_block, mrf_block)
            blocks.append(block)

            current_channels = current_channels >> 1

        self.blocks = nn.Sequential(*blocks)

        self.out_proj = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(current_channels, 1, 7, 
                      padding= (7 - 1) // 2),
            nn.Tanh()
        )

    def forward(self, mel):
        mel = self.in_proj(mel)
        mel = self.blocks(mel)
        audio = self.out_proj(mel)
        return {"pred_audio": audio}