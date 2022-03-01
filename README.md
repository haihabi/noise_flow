Noise Flow - A normalizing flows model for image noise modeling and synthesis
===
This is a fork of the NoiseFlow Original GitHub repository.
https://github.com/BorealisAI/noise_flow 

This repo includes the following: 
1. A PyTorch implementation of NoiseFlow
2. Parameter porting code (From Tensorflow to PyTorch)

The PyTorch Implementation is based on the repository https://github.com/haihabi/NormFlowPy.

# Original Readme

This repository provides the codes for training and testing the Noise Flow model used for image noise modeling and 
synthesis as described in the paper:

[**Noise Flow: Noise Modeling with Conditional Normalizing Flows**](https://arxiv.org/pdf/1908.08453.pdf)

It also provides code for training and testing a CNN-based image denoiser (DnCNN) using Noise Flow as a noise generator, with comparison to other noise generation methods (i.e., AWGN and signal-dependent noise).
  
# Required libraries

Python (works with 3.6)

TensorFlow (works with 1.12.0)

TensorFlow Probability (tested with 0.5.0)

_Despite not tested, the code may work with library versions other than the specified._

# Required dataset

[Smartphone Image Denoising Dataset (SIDD)](https://www.eecs.yorku.ca/~kamel/sidd/)

It is recommended to use the medium-size SIDD for training Noise Flow:

[SIDD_Medium_Raw](http://bit.ly/2kHT7Yr)

The code checks for and downloads `SIDD_Medium_Raw` if it does not exist. 

# Training/Testing/Sampling

Start by running `job_noise_flow.sh`

It contains a set of examples for training different models (as described in the paper) and optionally perform testing and 
sampling concurrently.

## Important parameters

`--sidd_path`: path to the SIDD dataset

`--arch`: the architecture of the noise flow model

`--cam`: (optional) to use/sample data from a specific camera

`--iso`: (optional) to use/sample data from a specific ISO level

Refer to `job_noise_flow.sh` or `ArgParser.py` for details on the rest of parameters.
   
# Sampling only

To use the Noise Flow trained model for generating noise samples:

Start by running `sample_noise_flow.py`

# Application to image denoising with DnCNN

Start by running `job_dncnn.sh` which contains examples for training DnCNN with synthetic noise from a Gaussian, signal-dependent, or Noise Flow model. 
Also, it contains an example for training with real noise from the SIDD.

# Paper

[Abdelrahman Abdelhamed](https://www.eecs.yorku.ca/~kamel/), [Marcus A. Brubaker](https://www.eecs.yorku.ca/~mab/), and [Michael S. Brown](https://www.eecs.yorku.ca/~mbrown/). Noise Flow: Noise Modeling with Conditional Normalizing Flows. In _ICCV_, 2019.

[PDF](https://arxiv.org/pdf/1908.08453.pdf)

# Citation

If you use Noise Flow in your research, we kindly ask that you cite the paper as follows:

    @inproceedings{abdelhamed2019noiseflow,
      title={{Noise Flow: Noise Modeling with Conditional Normalizing Flows}},
      author={Abdelhamed, Abdelrahman and Brubaker, Marcus A and Brown, Michael S},
      booktitle={International Conference on Computer Vision (ICCV)},
      year={2019}
    }

# License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
Public License. The terms and conditions can be found in the LICENSE file.

# Contact

[Abdelrahman Abdelhamed](https://www.eecs.yorku.ca/~kamel/) ([kamel@eecs.yorku.ca](mailto:kamel@eecs.yorku.ca))
