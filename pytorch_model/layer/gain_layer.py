import torch
import numpy as np
import normflowpy as nfp
from torch import nn


class GainLayer(nfp.ConditionalBaseFlowLayer):
    def __init__(self):
        super().__init__()
        self.gain_val = nn.Parameter(torch.randn(1))

    def forward(self, x, cond):
        n = np.prod(x.shape[1:])
        return x / self.gain_val, -n * torch.log(self.gain_val)

    def backward(self, z, cond):
        n = np.prod(z.shape[1:])
        return z * self.gain_val, n * torch.log(self.gain_val)
