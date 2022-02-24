import numpy as np
import torch
import normflowpy as nfp
import os
from pytorch_model.layer.signal_dependent_layer import SignalDependentLayer
from pytorch_model.layer.gain_layer import GainLayer
from pytorch_model.layer.noise_image_layer import ImageFlowStep
from torch.distributions import MultivariateNormal
from torch import nn


def unc_step(input_shape, n_channels, index, edge_bias, activation_function):
    class UNCAffine(nn.Module):
        def __init__(self, x_shape, n_outputs, width):
            super().__init__()
            self.net_class = nfp.base_nets.RealNVPConvBaseNet(x_shape, n_outputs, width,
                                                              activation_function=activation_function,
                                                              edge_bias=edge_bias)

        def forward(self, x):
            return self.net_class(x)

    return [nfp.flows.InvertibleConv2d1x1(n_channels),
            nfp.flows.AffineCoupling(input_shape, 0, net_class=UNCAffine)]


def get_activation_function(act_func):
    return nn.ReLU if act_func == "relu" else nn.SiLU


def generate_noise_flow(input_shape, device="cpu", edge_bias=True, activation_function="relu"):
    dim = int(np.prod(input_shape))
    n_channels = input_shape[0]
    activation_function = get_activation_function(activation_function)
    prior = MultivariateNormal(torch.zeros(dim, device=device),
                               torch.eye(dim, device=device))
    flows = [SignalDependentLayer()]
    i = 0
    for _ in range(4):
        flows.extend(unc_step(input_shape, n_channels=n_channels, index=i, edge_bias=edge_bias,
                              activation_function=activation_function))
        i += 1
    flows.append(GainLayer())
    for _ in range(4):
        flows.extend(unc_step(input_shape, n_channels=n_channels, index=i, edge_bias=edge_bias,
                              activation_function=activation_function))
        i += 1

    flows.append(nfp.flows.Tensor2Vector(input_shape))
    return nfp.NormalizingFlowModel(prior, flows)


def generate_noisy_image_flow(input_shape, device="cpu", load_model=False, edge_bias=True, activation_function="relu"):
    dim = int(np.prod(input_shape))
    n_channels = input_shape[0]
    activation_function = get_activation_function(activation_function)
    prior = MultivariateNormal(torch.zeros(dim, device=device),
                               torch.eye(dim, device=device))
    flows = [ImageFlowStep(), SignalDependentLayer()
             ]
    i = 0
    for _ in range(4):
        flows.extend(unc_step(input_shape, n_channels=n_channels, index=i, edge_bias=edge_bias,
                              activation_function=activation_function))
        i += 1
    flows.append(GainLayer())
    for _ in range(4):
        flows.extend(unc_step(input_shape, n_channels=n_channels, index=i, edge_bias=edge_bias,
                              activation_function=activation_function))
        i += 1

    flows.append(nfp.flows.Tensor2Vector(input_shape))
    flow = nfp.NormalizingFlowModel(prior, flows)
    if load_model:
        activation_name = "" if activation_function == "relu" else "silu"
        state_dict = torch.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "PyTorch", f"noisy_image_flow_{activation_name}.pt"),
            map_location=torch.device('cpu'))
        flow.load_state_dict(state_dict, strict=True)
    return flow
