from .scale_disc import ScaleDisctiminator
import torch.nn as nn

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale_discriminators = nn.ModuleList([
            ScaleDisctiminator(norm=True),
            ScaleDisctiminator(),
            ScaleDisctiminator()
        ])

        self.avgpool1 = nn.AvgPool1d(4, 2, padding=2)
        self.avgpool2 = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, generated, real):
        real_features = []
        generated_features = []
        real_outs = []
        generated_outs = []
        for i, discriminator in enumerate(self.scale_discriminators):
            if i == 0:
                r_in = real
                g_in = generated
            elif i == 1:
                r_in = self.avgpool1(real)
                g_in = self.avgpool1(generated)
            elif i == 2:
                r_in = self.avgpool2(self.avgpool1(real))
                g_in = self.avgpool2(self.avgpool1(generated))
            r_out, r_feat = discriminator(r_in)
            g_out, g_feat = discriminator(g_in)

            real_features.extend(r_feat)
            generated_features.extend(g_feat)
            real_outs.append(r_out)
            generated_outs.append(g_out)

        return real_outs, generated_outs, real_features, generated_features