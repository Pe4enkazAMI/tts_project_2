import torch.nn as nn
from .multi_period_descrominator import MultiPeriodDescriminator
from .multi_scale_disc import MultiScaleDiscriminator
import torch

class Descriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mpd = MultiPeriodDescriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, generated, real):
        if generated.shape[-1] > real.shape[-1]:
            pad_amount = generated.shape[-1] - real.shape[-1]
            pad = torch.zeros((real.shape[0], real.shape[1], pad_amount), device=real.device)
            real = torch.cat([real, pad], dim=-1)

        p_real_outs, p_gen_outs, p_real_feat, p_gen_feat = self.mpd(generated, real)
        s_real_outs, s_gen_outs,s_real_feat, s_gen_feat = self.msd(generated, real)
        return {
            "period_real_outs": p_real_outs,
            "period_gen_outs": p_gen_outs,
            "period_real_feat": p_real_feat,
            "period_gen_feat": p_gen_feat,
            "scale_real_outs": s_real_outs,
            "scale_gen_outs": s_gen_outs,
            "scale_real_feat": s_real_feat,
            "scale_gen_feat": s_gen_feat
        }
