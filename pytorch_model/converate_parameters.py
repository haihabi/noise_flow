import numpy as np
import torch
from pytorch_model.noise_flow import generate_noisy_image_flow, generate_noise_flow


def set_param_weigth(pytorch_param, numpy_param):
    pytorch_param.data = torch.tensor(numpy_param.astype("float32"), device=pytorch_param.device)


def set_parameters_conv(conv_module, w_np, b_np):
    w1_pytorch = np.transpose(w_np, (3, 2, 0, 1))
    set_param_weigth(conv_module.weight, w1_pytorch)
    set_param_weigth(conv_module.bias, b_np.flatten())


def set_parameters_bn(bn_module, mean_w, var_w):
    set_param_weigth(bn_module.running_mean, mean_w)
    set_param_weigth(bn_module.running_var, var_w)


def _sdn_layer_parameters_setter(flow_step, tf_parameters):
    cam_params = tf_parameters.pop('model/sdn_gain/cam_params')
    set_param_weigth(flow_step.cam_param, cam_params)

    gain_params = tf_parameters.pop('model/sdn_gain/gain_params')
    set_param_weigth(flow_step.gain_params, gain_params)

    beta1 = tf_parameters.pop('model/sdn_gain/beta1')
    set_param_weigth(flow_step.beta1, beta1)

    beta2 = tf_parameters.pop('model/sdn_gain/beta2')
    set_param_weigth(flow_step.beta2, beta2)

    _ = tf_parameters.pop('level0/bijector0/rescaling_scale0')
    pass


def _gain_parameters_setter(flow_step, tf_parameters):
    gain_val = tf_parameters.pop('model/sdn_gain/gain_val')
    _ = tf_parameters.pop('level0/bijector5/rescaling_scale0')
    set_param_weigth(flow_step.gain_val, gain_val)


def set_lower_matrix(l_vector, n):
    l = np.zeros([n, n])
    l[-1, :-1] = np.flip(l_vector[:n - 1])
    l[-2, :-2] = np.flip(l_vector[n:n + 2])
    l[-3, :-3] = l_vector[3]
    return l


def set_upper_matrix(l_vector, n):
    l = np.zeros([n, n])
    l[0, 1:] = l_vector[:n - 1]
    l[1, 2:] = l_vector[n:n + 2]
    l[2, 3:] = l_vector[3]
    return l


def _unc_layer_parameters_setter(flow_step_conv, flow_step_affine, tf_parameters, index, unc_count=0):
    p = tf_parameters.pop(f'level0/bijector{index}/Conv2d_1x1_{index}/P_matpar_lu_conv2d_1x1_{index}_0')
    n = p.shape[0]

    l_vector = tf_parameters.pop(
        f'level0/bijector{index}/Conv2d_1x1_{index}/L_vec_matpar_lu_conv2d_1x1_{index}_0')
    l = set_lower_matrix(l_vector, n)

    u_vector = tf_parameters.pop(
        f'level0/bijector{index}/Conv2d_1x1_{index}/U_vec_matpar_lu_conv2d_1x1_{index}_0')
    u = set_upper_matrix(u_vector, n)

    s_sign = tf_parameters.pop(f'level0/bijector{index}/Conv2d_1x1_{index}/sign_S_matpar_lu_conv2d_1x1_{index}_0')
    log_s = tf_parameters.pop(f'level0/bijector{index}/Conv2d_1x1_{index}/log_S_matpar_lu_conv2d_1x1_{index}_0')
    s = s_sign * np.exp(log_s)

    set_param_weigth(flow_step_conv.S, s)
    set_param_weigth(flow_step_conv.P, p)
    set_param_weigth(flow_step_conv.L, l)
    set_param_weigth(flow_step_conv.U, u)

    rescaling_scale = tf_parameters.pop(f'level0/bijector{index}/rescaling_scale0')
    set_param_weigth(flow_step_affine.s_cond.scale, rescaling_scale)

    base_name = "model/real_nvp_conv_template"
    if unc_count > 0:
        base_name += f"_{unc_count}"
    print(base_name)
    w1 = tf_parameters.pop(f"{base_name}/l_1/W")
    b1 = tf_parameters.pop(f"{base_name}/l_1/b")

    set_parameters_conv(flow_step_affine.s_cond.seq[0], w1, b1)
    w1 = tf_parameters.pop(f"{base_name}/l_2/W")
    b1 = tf_parameters.pop(f"{base_name}/l_2/b")
    set_parameters_conv(flow_step_affine.s_cond.seq[3], w1, b1)

    w1 = tf_parameters.pop(f"{base_name}/l_last/W")
    b1 = tf_parameters.pop(f"{base_name}/l_last/b")
    set_parameters_conv(flow_step_affine.s_cond.seq[7], w1, b1)

    mean = tf_parameters.pop(f"{base_name}/bn_nvp_conv_1/mean")
    var = tf_parameters.pop(f"{base_name}/bn_nvp_conv_1/var")
    set_parameters_bn(flow_step_affine.s_cond.seq[1], mean, var)

    mean = tf_parameters.pop(f"{base_name}/bn_nvp_conv_2/mean")
    var = tf_parameters.pop(f"{base_name}/bn_nvp_conv_2/var")
    set_parameters_bn(flow_step_affine.s_cond.seq[4], mean, var)

    logs = tf_parameters.pop(f"{base_name}/l_last/logs")
    set_param_weigth(flow_step_affine.s_cond.seq[8].logs, logs)


def converate_w(noise_flow, tf_parameters, shift=0):
    shift = int(len(noise_flow.flow.flows) == 20)
    _sdn_layer_parameters_setter(noise_flow.flow.flows[shift], tf_parameters)
    for i in range(4):
        _unc_layer_parameters_setter(noise_flow.flow.flows[1 + shift + 2 * i], noise_flow.flow.flows[2 + shift + 2 * i],
                                     tf_parameters,
                                     i + 1, 7 - i)

    _gain_parameters_setter(noise_flow.flow.flows[9 + shift], tf_parameters)
    for i in range(4):
        _unc_layer_parameters_setter(noise_flow.flow.flows[10 + shift + 2 * i],
                                     noise_flow.flow.flows[11 + shift + 2 * i],
                                     tf_parameters,
                                     i + 6, 3 - i)

    print("Finisihed Parametere Converstion")
