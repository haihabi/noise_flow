import glob
import logging
import os

import cv2
import torch
from scipy.io import savemat

from borealisflows.NoiseFlowWrapper import NoiseFlowWrapper
from mylogger import add_logging_level
import sidd.data_loader as loader
import pandas as pd
import numpy as np

from sidd.data_loader import check_download_sidd
from sidd.pipeline import process_sidd_image
from sidd.raw_utils import read_metadata
from sidd.sidd_utils import unpack_raw, kl_div_3_data
from pytorch_model.noise_flow import generate_noise_flow
from pytorch_model.converate_parameters import converate_w

data_dir = 'data'
sidd_path = os.path.join(data_dir, 'SIDD_Medium_Raw/Data')
nf_model_path = 'models/NoiseFlow'

samples_dir = os.path.join(data_dir, 'samples')
os.makedirs(samples_dir, exist_ok=True)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def compare_tensors(a, b):
    delta = np.power(a.cpu().detach().numpy() - np.transpose(b, (0, 3, 1, 2)), 2.0)
    error = delta.mean()
    if error > 1e-6:
        print(delta)
    print(error)
    return delta


def main():
    add_logging_level('TRACE', 100)
    logging.getLogger(__name__).setLevel("TRACE")
    logging.basicConfig(level=logging.TRACE)
    patch_size = 32
    save_model = False
    load_model = True
    # Prepare NoiseFlow
    # Issue: Low-probability sampling leading to synthesized pixels with too-high noise variance.
    # Solution: Contracting the sampling distribution by using sampling temperature less than 1.0 (e.g., 0.6).
    # Reference: Parmar, Niki, et al. "Image Transformer." ICML. 2018.
    noise_flow = NoiseFlowWrapper(nf_model_path, sampling_temperature=0.6)
    noise_flow_pytorch = generate_noise_flow([noise_flow.x_shape[-1], *noise_flow.x_shape[1:3]])
    if load_model:
        noise_flow_pytorch.load_state_dict(
            torch.load(f"/Users/haihabi/projects/noise_flow/models/PyTorch/noise_flow.pt",
                       map_location=torch.device('cpu')))
    else:
        import tensorflow as tf
        graph = tf.get_default_graph()
        parameters_data_dict = noise_flow.sess.run(
            {n.name: n.outputs[0] for n in graph._nodes_by_id.values() if n.type in ["VariableV2"]})

        converate_w(noise_flow_pytorch, parameters_data_dict)

    for _ in range(1):
        # load images
        noisy = loader.load_raw_image_packed(
            "/Users/haihabi/projects/noise_flow/data/0001_001_S6_00100_00060_3200_L_X/0001_NOISY_RAW_010.MAT")
        clean = loader.load_raw_image_packed(
            "/Users/haihabi/projects/noise_flow/data/0001_001_S6_00100_00060_3200_L_X/0001_GT_RAW_010.MAT")
        metadata, bayer_2by2, wb, cst2, iso, cam = read_metadata(
            "/Users/haihabi/projects/noise_flow/data/0001_001_S6_00100_00060_3200_L_X/0001_METADATA_RAW_010.MAT")
        iso = 100
        if iso not in [100, 400, 800, 1600, 3200]:
            continue

        np.random.seed(12345)  # for reproducibility
        n_pat = 10
        for p in range(n_pat):
            # crop patches
            v = np.random.randint(0, clean.shape[1] - patch_size)
            u = np.random.randint(0, clean.shape[2] - patch_size)

            clean_patch = clean[0, v:v + patch_size, u:u + patch_size, :]
            noisy_patch = noisy[0, v:v + patch_size, u:u + patch_size, :]

            clean_patch = np.expand_dims(clean_patch, 0)

            # sample noise
            noise_patch_syn, z, x_list = noise_flow.sample_noise_nf(clean_patch, 0.0, 0.0, iso, cam)

            ##########################
            # Sample Pytorch Mod
            ##########################
            clean_image_patch_pytorch = torch.Tensor(clean_patch)
            z_pytorch = torch.tensor(np.transpose(z, (0, 3, 1, 2)))
            noise_flow_pytorch.train()
            noise_patch_pytorch, logdet_pytorch = noise_flow_pytorch.backward(z_pytorch.reshape([1, -1]), cond=[
                torch.permute(clean_image_patch_pytorch, dims=(0, 3, 1, 2)), iso, cam])

            _ = compare_tensors(noise_patch_pytorch[1], z)
            _ = compare_tensors(noise_patch_pytorch[1], x_list[0])
            _ = compare_tensors(noise_patch_pytorch[2], x_list[1])
            _ = compare_tensors(noise_patch_pytorch[3], x_list[2])
            _ = compare_tensors(noise_patch_pytorch[4], x_list[3])
            _ = compare_tensors(noise_patch_pytorch[5], x_list[4])
            _ = compare_tensors(noise_patch_pytorch[6], x_list[5])
            _ = compare_tensors(noise_patch_pytorch[7], x_list[6])
            _ = compare_tensors(noise_patch_pytorch[8], x_list[7])
            _ = compare_tensors(noise_patch_pytorch[9], x_list[8])
            _ = compare_tensors(noise_patch_pytorch[10], x_list[9])
            _ = compare_tensors(noise_patch_pytorch[11], x_list[10])
            _ = compare_tensors(noise_patch_pytorch[12], x_list[11])
            _ = compare_tensors(noise_patch_pytorch[13], x_list[12])
            _ = compare_tensors(noise_patch_pytorch[14], x_list[13])
            _ = compare_tensors(noise_patch_pytorch[15], x_list[14])
            _ = compare_tensors(noise_patch_pytorch[16], x_list[15])
            _ = compare_tensors(noise_patch_pytorch[17], x_list[16])
            _ = compare_tensors(noise_patch_pytorch[18], x_list[17])
            _ = compare_tensors(noise_patch_pytorch[19], noise_patch_syn)
            if save_model:
                torch.save(noise_flow_pytorch.state_dict(),
                           os.path.join("/Users/haihabi/projects/noise_flow/models/PyTorch", "noise_flow.pt"))


if __name__ == '__main__':
    main()
