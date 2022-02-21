import numpy as np
import torch
import normflowpy as nfp
import os
from pytorch_model.layer.signal_dependent_layer import SignalDependentLayer
from pytorch_model.layer.gain_layer import GainLayer
from pytorch_model.layer.noise_image_layer import ImageFlowStep
from torch.distributions import MultivariateNormal


def unc_step(input_shape, n_channels, index):
    return [nfp.flows.InvertibleConv2d1x1(n_channels),
            nfp.flows.AffineCoupling(input_shape, 0)]


def generate_noise_flow(input_shape, device="cpu"):
    dim = int(np.prod(input_shape))
    n_channels = input_shape[0]
    prior = MultivariateNormal(torch.zeros(dim, device=device),
                               torch.eye(dim, device=device))
    flows = [SignalDependentLayer()]
    i = 0
    for _ in range(4):
        flows.extend(unc_step(input_shape, n_channels=n_channels, index=i))
        i += 1
    flows.append(GainLayer())
    for _ in range(4):
        flows.extend(unc_step(input_shape, n_channels=n_channels, index=i))
        i += 1

    flows.append(nfp.flows.Tensor2Vector(input_shape))
    return nfp.NormalizingFlowModel(prior, flows)


def generate_noisy_image_flow(input_shape, device="cpu", load_model=False):
    dim = int(np.prod(input_shape))
    n_channels = input_shape[0]
    prior = MultivariateNormal(torch.zeros(dim, device=device),
                               torch.eye(dim, device=device))
    flows = [ImageFlowStep(), SignalDependentLayer()
             ]
    i = 0
    for _ in range(4):
        flows.extend(unc_step(input_shape, n_channels=n_channels, index=i))
        i += 1
    flows.append(GainLayer())
    for _ in range(4):
        flows.extend(unc_step(input_shape, n_channels=n_channels, index=i))
        i += 1

    flows.append(nfp.flows.Tensor2Vector(input_shape))
    flow = nfp.NormalizingFlowModel(prior, flows)
    if load_model:
        state_dict = torch.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "PyTorch", "noisy_image_flow.pt"),
            map_location=torch.device('cpu'))
        flow.load_state_dict(state_dict,strict=True)
    return flow
