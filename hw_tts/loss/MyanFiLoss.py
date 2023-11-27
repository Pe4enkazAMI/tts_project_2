import torch 
import torch.nn as nn
from hw_tts.preproc import MelSpectrogram

class GenLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 2
        self.beta = 45
        self.l1loss = nn.L1Loss()
        self.mel_spec = MelSpectrogram()

    def forward(self,
                spectrogram, 
                pred_audio,
                period_gen_outs,
                period_real_feat,
                period_gen_feat,
                scale_gen_outs,
                scale_real_feat,
                scale_gen_feat,
                **kwargs):
        
        generated_audio = pred_audio.squeeze(1) # remove channel
        generated_spectrogram = self.mel_spec(generated_audio) 
        
        if spectrogram.shape[-1] < generated_spectrogram.shape[-1]:
            diff = generated_spectrogram.shape[-1] - spectrogram.shape[-1]
            pad_value = self.mel_spec.config.pad_value
            pad = torch.zeros((spectrogram.shape[0], spectrogram.shape[1], diff))
            pad = pad.fill_(pad_value).to(spectrogram.device)
            spectrogram = torch.cat([spectrogram, pad], dim=-1)

        # adv_loss
        adv_loss = 0
        for p in period_gen_outs:
            adv_loss = adv_loss + torch.mean((p - 1) ** 2)
        for s in scale_gen_outs:
            adv_loss = adv_loss + torch.mean((s - 1) ** 2)

        # fm_loss
        fm_loss = 0
        for real, gen in zip(period_real_feat, period_gen_feat):
            fm_loss = fm_loss + self.l1loss(gen, real)
        for real, gen in zip(scale_real_feat, scale_gen_feat):
            fm_loss = fm_loss + self.l1loss(gen, real)

        # mel_loss
        mel_loss = self.l1loss(generated_spectrogram, spectrogram)

        GenLoss = adv_loss + self.alpha * fm_loss + self.beta * mel_loss

        return GenLoss, adv_loss, fm_loss, mel_loss
        

class DescLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
        period_real_outs,
        period_gen_outs,
        scale_real_outs,
        scale_gen_outs,
        **kwargs):

        # D_loss
        DescLoss = 0
        for period_real, period_gen in zip(period_real_outs, period_gen_outs):
            DescLoss = DescLoss + torch.mean((period_real - 1) ** 2) + torch.mean((period_gen - 0) ** 2)
        for scale_real, scale_gen in zip(scale_real_outs, scale_gen_outs):
            DescLoss = DescLoss + torch.mean((scale_real - 1) ** 2) + torch.mean((scale_gen - 0) ** 2)

        return DescLoss
                