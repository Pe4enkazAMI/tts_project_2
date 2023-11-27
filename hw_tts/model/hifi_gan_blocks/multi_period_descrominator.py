import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .period_descrominator import PeriodDescriminator


class MultiPeriodDescriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([ PeriodDescriminator(self.periods[0]),
                                              PeriodDescriminator(self.periods[1]),
                                              PeriodDescriminator(self.periods[2]),
                                              PeriodDescriminator(self.periods[3]),
                                              PeriodDescriminator(self.periods[4]),
                                            ])
        
    def forward(self, gen, gt):
        gen_ft, gen_sc = [], []
        gt_ft, gt_sc = [], []
        for disc in self.discriminators:
            gt_out, gt_feat = disc(gt)
            gen_out, gen_feat = disc(gen)
            gt_ft.extend(gt_feat)
            gen_ft.extend(gen_feat)
            gt_sc.append(gt_out)
            gen_sc.append(gen_out)
        return gt_sc, gen_sc, gt_ft, gen_ft