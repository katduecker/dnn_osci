# %% [code]
# %% [code]
import numpy as np
import torch
import aet_net_2lay  # 2-layer architecture
from aet_net import sigmoid


# forward dynamics/discretization, euler integration single neuron
def euler_dyn(Z, t_, params_, alpha_params_, sig_param, t_start=0):

    # inputs:
    # input_: input image
    # params_: hyperparameters
    # t_: time vector for integration
    # alpha_params: alpha frequency & amplitude

    # discretization & dynamics parameters
    tau_h, tau_R, c, S = params_

    # alpha frequency & amplitude
    _af, _aa, _ap = alpha_params_

    # initialize empty matrices
    dt = np.diff(t_)[0]
    dh1dt = np.zeros((len(t_) + 1,))
    dR1dt = np.zeros((len(t_) + 1,))
    dZdt = np.zeros((len(t_) + 1,))

    dR1dt[0] = (c / (c - 1)) * Z
    dh1dt[0] = Z / (c - 1)

    # alpha inhibition
    alpha_inh1 = _aa * np.sin(2 * np.pi * _af * t_ + _ap) + _aa

    boxcar = np.zeros(len(t_))
    boxcar[t_start:] = 1

    for _it, t in enumerate(t_):

        # pre-activation
        dZdt[_it + 1] = (
            Z * boxcar[_it] + dh1dt[_it] - dR1dt[_it] - alpha_inh1[_it]
        ) / S

        # dynamics hidden layer1
        dh1dt[_it + 1] = dh1dt[_it] + dt / tau_h * (
            -dh1dt[_it] + sigmoid(dZdt[_it + 1], sig_param)
        )

        # adaptation term 1
        dR1dt[_it + 1] = dR1dt[_it] + dt / tau_R * (-dR1dt[_it] + c * dh1dt[_it + 1])

    return dZdt, dh1dt, dR1dt


# 2-layer NN Euler

# scaled c to each Z


def euler_dyn_2layer(
    model, input_, params_, t_, alpha_params, DEVICE, inp_on, start_fix=True
):

    # inputs:
    # input_: input image
    # params_: hyperparameters
    # t_: time vector for integration
    # alpha_params: alpha frequency & amplitude

    # discretization & dynamics parameters
    tau_h, tau_R, c, S = params_

    # alpha frequency & amplitude
    _af, _aa, _aph = alpha_params

    with torch.no_grad():

        # preactivation (dot product of input and first weight matrix)
        Z, H, _, _ = model.forw_conv(input_)
        Z2 = model.fc1(H)

        # initialize empty matrices
        dt = np.diff(t_)[0]
        dh1dt = (torch.ones((model.dims[1], len(t_) + 1))).to(DEVICE)
        dR1dt = (torch.zeros((model.dims[1], len(t_) + 1))).to(DEVICE)

        dh2dt = (torch.zeros((model.dims[2], len(t_) + 1))).to(DEVICE)
        dR2dt = (torch.zeros((model.dims[2], len(t_) + 1))).to(DEVICE)

        if start_fix != 0:

            for i, z in enumerate(Z):

                dR1dt[i, 0] = (c / (c - 1)) * start_fix
                dh1dt[i, 0] = start_fix / (c - 1)

            for i, z in enumerate(Z2):
                dR2dt[i, 0] = (c / (c - 1)) * start_fix
                dh2dt[i, 0] = start_fix / (c - 1)

        dOdt = (torch.zeros((model.dims[3], len(t_) + 1))).to(DEVICE)
        dZdt = (torch.zeros((model.dims[1], len(t_) + 1))).to(DEVICE)
        dZ2dt = (torch.ones((model.dims[2], len(t_) + 1))).to(DEVICE)

        # alpha inhibition
        alpha_inh1 = _aa[0] * np.sin(2 * np.pi * _af[0] * t_ + _aph[0]) + _aa[0]
        alpha_inh2 = _aa[1] * np.sin(2 * np.pi * _af[1] * t_ + _aph[1]) + _aa[1]

        # create boxcar function to try different input onsets
        boxcar = np.zeros_like(t_)
        boxcar[inp_on:] = 1

        for _it, t in enumerate(t_):

            # dynamic input: multiply input with boxcar
            Z, H1, H2, O = model.forw_conv(input_ * boxcar[_it])

            # pre-activation
            dZdt[:, _it + 1] = (Z + dh1dt[:, _it] - dR1dt[:, _it] - alpha_inh1[_it]) / S

            # dynamics hidden layer
            dh1dt[:, _it + 1] = dh1dt[:, _it] + dt / tau_h * (
                -dh1dt[:, _it] + model.acti1(dZdt[:, _it + 1], model.sig_param[0])
            )

            # adaptation term
            dR1dt[:, _it + 1] = dR1dt[:, _it] + dt / tau_R * (
                -dR1dt[:, _it] + c * dh1dt[:, _it + 1]
            )

            # pre-activation layer 2
            Z2 = model.fc1(H1)

            dZ2dt[:, _it + 1] = (
                model.fc1(dh1dt[:, _it + 1])
                + dh2dt[:, _it + 1]
                - dR2dt[:, _it]
                - alpha_inh2[_it]
            ) / S
            # dynamics hidden layer1
            dh2dt[:, _it + 1] = dh2dt[:, _it] + dt / tau_h * (
                -dh2dt[:, _it]
                + aet_net_2lay.sigmoid(dZ2dt[:, _it + 1], model.sig_param[1])
            )

            # # adaptation term 2
            dR2dt[:, _it + 1] = dR2dt[:, _it] + dt / tau_R * (
                -dR2dt[:, _it] + c * dh2dt[:, _it + 1]
            )

            # output layer
            dOdt[:, _it + 1] = model.actiout(model.fc2(dh2dt[:, _it + 1]))

    return dZdt, dZ2dt, dh1dt, dR1dt, dh2dt, dR2dt, dOdt


# approximate fixed point
def fun_fixed_point_num(x, Z, c, S):
    return (Z / (c - 1)) + (S / (c - 1)) * (np.log(1 / x - 1) / 2 - 2.5) - x
