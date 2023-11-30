import torch
from hw_tts.model.hifi_gan_blocks.descriminator import Descriminator
from hw_tts.model.hifi_gan_blocks.generator import Generator
from torch import nn
import numpy as np


class MyanFi(nn.Module):
    def __init__(self, 
                 input_channels, 
                 hidden_channels, 
                 upsample_kernels,
                 upsample_stride, 
                 resblock_kernels,
                 resblock_dilations):
        super().__init__()
        
        self.generator = Generator(input_channels,
                                   hidden_channels,
                                   upsample_kernels,
                                   upsample_stride, 
                                   resblock_kernels,
                                   resblock_dilations)
        self.descriminator = Descriminator()

    def forward(self, spectrogram, *args, **kwargs):
        return self.generator(spectrogram)

    def generate(self, **batch):
        return self.forward(**batch)

    def _descriminator(self, generated, real):
        return self.descriminator(generated, real)
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)