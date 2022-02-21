import torch

import normflowpy as nfp
from torch import nn

ISO2INDEX = {100: 0,
             400: 1,
             800: 2,
             1600: 3,
             3200: 4}


class SignalDependentLayer(nfp.ConditionalBaseFlowLayer):
    def __init__(self, m_iso=5, n_cam=5):
        super().__init__()
        self.cam_param = nn.Parameter(torch.ones(3, n_cam))
        self.gain_params = nn.Parameter(torch.ones(m_iso))
        self.beta1 = nn.Parameter(torch.ones(1))
        self.beta2 = nn.Parameter(torch.ones(1))

    def _build_scale(self, clean_image, iso, cam):
        one_cam_params = self.cam_param[:, cam]
        one_cam_params = torch.exp(one_cam_params)
        gain_index = ISO2INDEX[iso.item()]
        g = self.gain_params[gain_index]
        gain = torch.exp(g * one_cam_params[2]) * iso
        beta1 = torch.exp(self.beta1 * one_cam_params[0])
        beta2 = torch.exp(self.beta2 * one_cam_params[1])
        return torch.sqrt(beta1 * clean_image / gain + beta2)

    def forward(self, x, cond):
        clean_image = cond[0]
        iso = cond[1]
        cam = cond[2]
        scale = self._build_scale(clean_image, iso, cam)
        return x / scale, -torch.sum(torch.log(scale), dim=[1, 2, 3])

    def backward(self, z, cond):
        clean_image = cond[0]
        iso = cond[1]
        cam = cond[2]
        scale = self._build_scale(clean_image, iso, cam)
        return z * scale, torch.sum(torch.log(scale), dim=[1, 2, 3])
