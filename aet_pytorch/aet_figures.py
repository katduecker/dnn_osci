import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
from matplotlib.patches import Rectangle

import numpy as np
from itertools import combinations
import scipy

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# scientific colormap
#!pip install cmcrameri;
from cmcrameri import cm

lpcm = cm.batlowS.colors[[0, 4, 5], :]

# integration function
from aet_dyn import euler_dyn, euler_dyn_2layer
import aet_net_2lay  # 2-layer architecture
from aet_net import sigmoid


def fig1():

    timevec = np.linspace(0, 1, 1000)  # time vector

    # font type and size
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["figure.figsize"] = (7, 3)

    stim1 = 0.5 * np.sin(2 * np.pi * timevec * 20) + 0.5  # passionfruit
    stim2 = 0.45 * np.sin(2 * np.pi * timevec * 20 - np.pi / 2) + 0.45  # apple

    # inhibition
    alpha = (
        0.5 * np.sin(2 * np.pi * timevec * 10 + np.pi + np.pi / 3 - np.pi / 100 * 10)
        + 0.5
    )  # used to modulate actual values
    alpha_plot = (
        0.5 * np.cos(2 * np.pi * timevec * 10 + np.pi + np.pi / 3 - np.pi / 100 * 10)
        + 0.5
    )  # used for plotting

    s1 = stim1 * alpha
    s11 = s1
    s2 = stim2 * alpha
    s21 = s2

    fig, axs = plt.subplots(2, 1)

    # plot stimulus activations and alpha oscillation
    axs[0].plot(timevec[: s11.shape[0]], s11, linewidth=3)
    axs[0].plot(timevec[: s21.shape[0]], s21, linewidth=3)
    axs[0].plot(
        timevec[: s11.shape[0]],
        alpha_plot[: s11.shape[0]],
        color=[0.25, 0.25, 0.25],
        linewidth=1,
    )

    axs[0].axis("off")

    # time points representation both stimuli
    axs[0].plot(
        np.tile(0.0728, (100,)),
        np.linspace(-0.1, 1, 100),
        color=[0.25, 0.25, 0.25],
        linestyle=":",
        linewidth=1,
    )
    axs[0].plot(
        np.tile(0.1728, (100,)),
        np.linspace(-0.1, 1, 100),
        color=[0.25, 0.25, 0.25],
        linestyle=":",
        linewidth=1,
    )
    axs[0].plot(
        np.tile(0.2728, (100,)),
        np.linspace(-0.1, 1, 100),
        color=[0.25, 0.25, 0.25],
        linestyle=":",
        linewidth=1,
    )
    axs[0].plot(
        np.tile(0.3728, (100,)),
        np.linspace(-0.1, 1, 100),
        color=[0.25, 0.25, 0.25],
        linestyle=":",
        linewidth=1,
    )

    axs[0].set_xlim((0, 0.4))
    axs[0].set_ylim((-0.1, 1.1))

    axs[0].set_title("feature-selective cortex", loc="left")

    # repeat for object-selective cortex
    s12 = np.concatenate(
        (
            np.zeros(
                10,
            ),
            s11[:-10],
        )
    )
    s22 = np.concatenate(
        (
            np.zeros(
                10,
            ),
            s21[:-10],
        )
    )

    axs[1].plot(timevec[: s12.shape[0]], s12, linewidth=3)
    axs[1].plot(timevec[: s22.shape[0]], s22, linewidth=3)
    axs[1].axis("off")
    axs[1].set_title("object-selective cortex", loc="left")

    axs[1].plot(
        np.tile(0.0728, (110,)),
        np.linspace(-0.1, 1.1, 110),
        color=[0.25, 0.25, 0.25],
        linestyle=":",
        linewidth=1,
    )
    axs[1].plot(
        np.tile(0.1728, (110,)),
        np.linspace(-0.1, 1.1, 110),
        color=[0.25, 0.25, 0.25],
        linestyle=":",
        linewidth=1,
    )
    axs[1].plot(
        np.tile(0.2728, (110,)),
        np.linspace(-0.1, 1.1, 110),
        color=[0.25, 0.25, 0.25],
        linestyle=":",
        linewidth=1,
    )
    axs[1].plot(
        np.tile(0.3728, (110,)),
        np.linspace(-0.1, 1.1, 110),
        color=[0.25, 0.25, 0.25],
        linestyle=":",
        linewidth=1,
    )
    alpha_plot = (
        0.5 * np.cos(2 * np.pi * timevec * 10 + np.pi + np.pi / 3 - np.pi / 100 * 30)
        + 0.5
    )

    axs[1].plot(
        timevec[: s22.shape[0]],
        alpha_plot[: s22.shape[0]],
        color=[0.25, 0.25, 0.25],
        linewidth=1,
    )
    axs[1].set_xlim((0, 0.4))
    axs[1].set_ylim((-0.1, 1.1))

    fig.tight_layout(pad=2)

    return fig


def fig2(loss_hist2, idx, label, H21, H22, O2):

    plt.rcParams["svg.fonttype"] = "none"

    fig = plt.figure(figsize=(12, 10))

    # set up layout
    gs = GridSpec(5, 9, figure=fig, height_ratios=[1, 0.75, 0.75, 0.75, 0.1])
    ax1 = fig.add_subplot(gs[0, 3:6])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 2:7])
    ax4 = fig.add_subplot(gs[3, 4:5])
    ax5 = fig.add_subplot(gs[4, :])

    ax1.spines[["right", "top"]].set_visible(False)
    ax1.set_xlabel("epoch")
    y_max = torch.round(torch.max(loss_hist2), decimals=2)
    ax1.set_yticks((0, y_max / 2, y_max))
    ax1.set_xticks(np.arange(0, len(loss_hist2) + 5, 5))
    ax1.set_xlim((0, len(loss_hist2)))

    ax1.set_ylabel("loss")
    # plot loss
    ax1.plot(np.arange(loss_hist2.shape[0]), loss_hist2, color="k", linewidth=2)

    # plot activation H1
    ax2.imshow(
        H21[idx].detach().cpu().numpy(),
        cmap=cm.lajolla_r,
        interpolation="nearest",
        aspect=1,
    )
    ax2.set_xticks(np.arange(0, 64, 7))
    ax2.set_xticklabels(np.arange(1, 65, 7))
    ax2.set_yticks(np.arange(3))
    ax2.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax2.set_xticks(np.arange(-0.5, 65, 1), minor=True)
    ax2.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax2.tick_params(which="minor", bottom=False, left=False)
    ax2.set_yticklabels(["A", "E", "T"])
    ax2.spines[:].set_visible(False)

    # plot activation H2
    ax3.imshow(
        H22[idx].detach().cpu().numpy(),
        cmap=cm.lajolla_r,
        interpolation="nearest",
        aspect=1,
    )
    ax3.set_xticks(np.arange(0, 32, 5))
    ax3.set_xticklabels(np.arange(1, 32, 5))

    ax3.set_yticks(np.arange(3))
    ax3.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax3.set_xticks(np.arange(-0.5, 32, 1), minor=True)
    ax3.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax3.tick_params(which="minor", bottom=False, left=False)

    ax3.spines[:].set_visible(False)

    ax3.set_yticks(np.arange(3))
    ax3.set_yticklabels(["A", "E", "T"])
    ax3.set_xlabel("hidden unit")

    # plot actiation output
    im4 = ax4.imshow(
        O2[idx].detach().cpu().numpy(), cm.lajolla_r, interpolation="nearest", aspect=1
    )
    ax4.set_xticks(np.arange(0, 3, 1))
    ax4.set_xticklabels(np.arange(3))

    ax4.set_yticks(np.arange(3))
    ax4.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax4.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax4.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax4.tick_params(which="minor", bottom=False, left=False)

    ax4.spines[:].set_visible(False)

    ax4.set_yticks(np.arange(3))
    ax4.set_yticklabels(["A", "E", "T"])
    ax4.set_xlabel("output unit")
    ax4.set_xlim([-0.5, 2.5])
    ax5.axis("off")
    im4.set_clim(0, 1)
    cb = fig.colorbar(im4, ax=ax5, orientation="horizontal", fraction=0.75)
    cb.set_label("activation")
    cb.set_ticks((0, 0.5, 1))

    fig.tight_layout()

    return fig


def fig3(maxZ, Z_vec, c_vec, S_vec, params, alpha_params_off, alpha_params_on):

    # prepare colormaps and layout
    step_cm = int(len(cm.lajolla_r.colors) / (maxZ+1))
    cmz = cm.lajolla_r.colors[0 : len(cm.lajolla_r.colors) : step_cm]
    cmz_map = matplotlib.colors.ListedColormap(cmz)
    lplot_c = cm.glasgowS.colors[3:6]
    fig, ax = plt.subplots()

    imfig = ax.imshow(np.array([np.arange(0, maxZ + 1, 0.5)]), cmap=cmz_map)

    plt.close(fig)

    # prepare layout
    fig, axs = plt.subplots(2, 2)
    plt.close(fig)

    tau_h, tau_R, c, S = params

    # time
    timevec = np.linspace(0, 1, 1000)
    t_start = 0

    # alpha parameters
    alpha_params = [10, 0, 0]
    sig_param = [2, -2.5]

    # Layout grid

    plt.rcParams["figure.figsize"] = (17, 8)

    fig = plt.figure(figsize=(16, 8))

    gs0 = fig.add_gridspec(2, 1, height_ratios=[0.75, 1])
    gs1 = gs0[0].subgridspec(2, 2, height_ratios=[1, 0.3])
    gs2 = gs0[1].subgridspec(3, 2, hspace=1, height_ratios=[1, 1, 0.25])
    gs200 = gs2[0, 0].subgridspec(1, 2, width_ratios=[1, 0.75])
    gs210 = gs2[1, 0].subgridspec(1, 2, width_ratios=[1, 0.75])
    gs201 = gs2[0, 1].subgridspec(1, 2, width_ratios=[1, 0.75])
    gs211 = gs2[1, 1].subgridspec(1, 2, width_ratios=[1, 0.75])
    gscb = gs2[2, 1].subgridspec(1, 2, width_ratios=[1, 0.75])

    # dynamis without alpha
    Zt, Ht, Rt = euler_dyn(
        maxZ, timevec, params, alpha_params_off, sig_param, t_start=0
    )

    peaks_r, _ = scipy.signal.find_peaks(Rt)
    troughs_r, _ = scipy.signal.find_peaks(-Rt)

    # dynamics with alpha
    Zta, Hta, Rta = euler_dyn(
        maxZ, timevec, params, alpha_params_on, sig_param, t_start=0
    )

    axs[0, 0] = fig.add_subplot(gs1[0, 0])

    axs[0, 0].plot(timevec, Ht[t_start + 1 :], color=lplot_c[-1], linewidth=3)
    axs[0, 0].set_ylim((0, 1.1))
    axs[0, 0].set_yticks((0, 0.5, 1))
    axs[0, 0].set_ylabel("H")
    axs[0, 0].spines[["top"]].set_visible(False)
    ax2 = axs[0, 0].twinx()
    ax2.plot(timevec, Rt[t_start + 1 :], color=lplot_c[1], linewidth=3)

    max_R = np.ceil(np.max(Rt))
    ax2.set_ylim((0, max_R + max_R * 0.1))
    ax2.set_yticks((0, int(max_R) / 2, int(max_R)))
    ax2.set_ylabel("R")
    ax2.set_xlabel("time (s)")
    ax2.spines[["top"]].set_visible(False)

    l = [None] * 3
    axs[0, 1] = fig.add_subplot(gs1[0, 1])
    (l[0],) = axs[0, 1].plot(
        timevec, Hta[t_start + 1 :], color=lplot_c[-1], linewidth=3
    )
    axs[0, 1].set_ylim((0, 1.1))
    axs[0, 1].set_yticks((0, 0.5, 1))
    axs[0, 1].set_ylabel("H")
    axs[0, 1].spines[["top"]].set_visible(False)

    (l[2],) = axs[0, 1].plot(
        timevec,
        0.5 * np.sin(2 * np.pi * 10 * timevec) + 0.5,
        color=lplot_c[0],
        linestyle="-.",
        linewidth=3,
    )

    ax2 = axs[0, 1].twinx()
    (l[1],) = ax2.plot(
        timevec, Rta[t_start + 1 :], color=lplot_c[1], alpha=0.5, linewidth=3
    )
    ax2.set_ylim((0, max_R + max_R * 0.1))
    ax2.set_yticks((0, int(max_R) / 2, int(max_R)))
    ax2.set_ylabel("R")
    ax2.set_xlabel("time (s)")
    ax2.spines[["top"]].set_visible(False)

    axs[1, 1] = fig.add_subplot(gs1[1, 1])
    axs[1, 1].axis("off")
    axs[1, 1].legend(
        l, ["H(t)", "R(t)", "alpha"], loc="lower left", frameon=False, ncol=5
    )

    # Scatters
    ## plot activations in nodes
    subax10 = fig.add_subplot(gs200[0])
    subax20 = fig.add_subplot(gs210[0])

    ## FREQUENCY as a function of C for different inputs Z
    for i, c in enumerate(c_vec):
        for iz, Z in enumerate(Z_vec):
            params = [tau_h, tau_R, c, S]
            Zt, Ht, Rt = euler_dyn(
                Z, timevec, params, alpha_params_off, sig_param, t_start=0
            )

            peaks, _ = scipy.signal.find_peaks(Ht)

            freq = np.mean(1 / (np.diff(peaks[2:]) / 1000))
            pow = np.mean(Ht[peaks[2:]])

            subax10.scatter(c, freq, color=cmz[iz], s=50)
            subax20.scatter(c, pow, color=cmz[iz], s=50)

    subax10.add_patch(
        Rectangle(
            (10.3, 4),
            0.5,
            16,
            edgecolor="k",
            facecolor="none",
            linewidth=2,
            linestyle=":",
        )
    )

    subax20.add_patch(
        Rectangle(
            (10.3, 4),
            0.5,
            16,
            edgecolor="k",
            facecolor="none",
            linewidth=2,
            linestyle=":",
        )
    )

    subax10.set_ylim((4, 20))
    subax10.set_ylabel("frequency (Hz)")
    subax10.set_xlabel("c")
    subax10.set_xlim((c_vec[0], c_vec[-1]))
    subax10.set_xticks(np.arange(c_vec[0], c_vec[-1] + 1, 3))
    subax10.set_yticks((4, 12, 20))
    subax10.spines[["top", "right"]].set_visible(False)

    subax20.set_ylim((0, 1.05))
    subax20.set_ylabel("amplitude")
    subax20.set_xlabel("c")
    subax20.set_xlim((c_vec[0], c_vec[-1]))
    subax20.set_xticks(np.arange(c_vec[0], c_vec[-1] + 1, 3))
    subax20.set_yticks((0, 0.5, 1))
    subax20.spines[["top", "right"]].set_visible(False)

    subax11 = fig.add_subplot(gs200[1])
    c = 10
    for iz, Z in enumerate(Z_vec):
        params = [tau_h, tau_R, c, S]
        Zt, Ht, Rt = euler_dyn(
            Z, timevec, params, alpha_params_off, sig_param, t_start=0
        )

        peaks, _ = scipy.signal.find_peaks(Ht)

        freq = np.mean(1 / (np.diff(peaks[2:]) / 1000))
        subax11.scatter(Z, freq, color=cmz[iz], s=50, edgecolor="k")

    subax11.set_ylim((4, 20))
    subax11.set_ylabel("frequency (Hz)")
    subax11.set_xlabel("Z")
    subax11.set_xlim((0, np.round(maxZ+1)))
    subax11.set_xticks(np.arange(0, maxZ + 1, (maxZ + 0.5) / 2))
    subax11.set_yticks((4, 12, 20))
    subax11.spines[["top", "right"]].set_visible(False)

    ## S
    subax21 = fig.add_subplot(gs210[1])
    c = 10

    for i, S in enumerate(S_vec):
        for iz, Z in enumerate(Z_vec):
            params = [tau_h, tau_R, c, S]
            Zt, Ht, Rt = euler_dyn(
                Z, timevec, params, alpha_params_off, sig_param, t_start=0
            )

            peaks, _ = scipy.signal.find_peaks(Ht)

            freq = np.mean(1 / (np.diff(peaks[2:]) / 1000))
            pow = np.mean(Ht[peaks[2:]])

            subax21.scatter(S, freq, color=cmz[iz], s=50)

    subax21.set_ylim((5, 15))
    subax21.set_ylabel("frequency (Hz)")
    subax21.set_xlabel("S")
    subax21.set_xlim((0, 0.1))
    subax21.set_xticks(np.arange(0, 0.11, 0.05))
    subax21.set_yticks((5, 10, 15))
    subax21.spines[["top", "right"]].set_visible(False)

    ## ALPHA  DYNAMICS

    # Scatters
    ## plot activations in nodes
    ## plot activations in nodes
    subax12 = fig.add_subplot(gs201[0])
    subax22 = fig.add_subplot(gs211[0])

    # frequency as a function of C for different inputs Z

    ## FREQUENCY
    for i, c in enumerate(c_vec):
        for iz, Z in enumerate(Z_vec):
            params = [tau_h, tau_R, c, S]
            Zt, Ht, Rt = euler_dyn(
                Z, timevec, params, alpha_params_on, sig_param, t_start=0
            )

            peaks, _ = scipy.signal.find_peaks(Ht)

            freq = np.mean(1 / (np.diff(peaks[2:]) / 1000))
            pow = np.mean(Ht[peaks[2:]])

            subax12.scatter(c, freq, color=cmz[iz], s=50)
            subax22.scatter(c, pow, color=cmz[iz], s=50)

    subax12.add_patch(
        Rectangle(
            (10.3, 4),
            0.5,
            16,
            edgecolor="k",
            facecolor="none",
            linewidth=2,
            linestyle=":",
        )
    )

    subax22.add_patch(
        Rectangle(
            (10.3, 4),
            0.5,
            16,
            edgecolor="k",
            facecolor="none",
            linewidth=2,
            linestyle=":",
        )
    )
    subax12.set_ylim((4, 20))
    subax12.set_ylabel("frequency (Hz)")
    subax12.set_xlabel("c")
    subax12.set_xlim((c_vec[0], c_vec[-1]))
    subax12.set_xticks(np.arange(c_vec[0], c_vec[-1] + 1, 3))
    subax12.set_yticks((4, 12, 20))
    subax12.spines[["top", "right"]].set_visible(False)

    subax22.set_ylim((0, 1.05))
    subax22.set_ylabel("amplitude")
    subax22.set_xlabel("c")
    subax22.set_xlim((c_vec[0], c_vec[-1]))
    subax22.set_xticks(np.arange(c_vec[0], c_vec[-1] + 1, 3))
    subax22.set_yticks((0, 0.5, 1))
    subax22.spines[["top", "right"]].set_visible(False)

    subax13 = fig.add_subplot(gs201[1])
    c = 10
    for iz, Z in enumerate(Z_vec):
        params = [tau_h, tau_R, c, S]
        Zt, Ht, Rt = euler_dyn(
            Z, timevec, params, alpha_params_on, sig_param, t_start=0
        )

        peaks, _ = scipy.signal.find_peaks(Ht)

        freq = np.mean(1 / (np.diff(peaks[2:]) / 1000))
        subax13.scatter(Z, freq, color=cmz[iz], s=50, edgecolor="k")

    subax13.set_ylim((4, 20))
    subax13.set_ylabel("frequency (Hz)")
    subax13.set_xlabel("Z")
    subax13.set_xlim((0, np.round(maxZ+1)))
    subax13.set_xticks(np.arange(0, maxZ + 1, (maxZ + 0.5) / 2))
    subax13.set_yticks((4, 12, 20))
    subax13.spines[["top", "right"]].set_visible(False)

    ## S
    subax23 = fig.add_subplot(gs211[1])
    c = 10

    for i, S in enumerate(S_vec):
        for iz, Z in enumerate(Z_vec):
            params = [tau_h, tau_R, c, S]
            Zt, Ht, Rt = euler_dyn(
                Z, timevec, params, alpha_params_on, sig_param, t_start=0
            )

            peaks, _ = scipy.signal.find_peaks(Ht)

            freq = np.mean(1 / (np.diff(peaks[2:]) / 1000))
            pow = np.mean(Ht[peaks[2:]])

            subax23.scatter(S, freq, color=cmz[iz], s=50)

    subax23.set_ylim((5, 15))
    subax23.set_ylabel("frequency (Hz)")
    subax23.set_xlabel("S")
    subax23.set_xlim((0, 0.1))
    subax23.set_xticks(np.arange(0, 0.11, 0.05))
    subax23.set_yticks((5, 10, 15))
    subax23.spines[["top", "right"]].set_visible(False)

    subcb = fig.add_subplot(gscb[1])
    subcb.axis("off")
    cb = fig.colorbar(imfig, ax=subcb, orientation="horizontal", fraction=1)
    cb.set_label("Z")
    cb.set_ticks(np.arange(0, maxZ + 1, (maxZ + 0.5) / 2))

    fig.tight_layout()

    return fig, peaks_r, troughs_r


def fig4(
    timevec,
    alpha_params,
    t_start,
    x_train,
    model2,
    Z21,
    Z22,
    H1t,
    Hstar1,
    H1ta,
    H2t,
    H2ta,
    Hstar2,
    R1t,
    Rstar1,
    R1ta,
    R2t,
    R2ta,
    Rstar2,
    Ot,
    Ota,
):

    # get all Z's for all inputs
    Zall = model2.forw_conv(x_train)[0]

    # find maximum
    maxZ = np.round(torch.max(Zall[:]).detach().numpy())

    # prepare colormaps and layout
    step_cm = int(len(cm.lajolla_r.colors) / maxZ)
    cmz = cm.lajolla_r.colors[0 : len(cm.lajolla_r.colors) : step_cm]
    cmz_map = matplotlib.colors.ListedColormap(cmz)
    lplot_c = cm.glasgowS.colors[3:6]

    fig, ax = plt.subplots()

    imfig = ax.imshow(np.array([np.arange(0, maxZ + 1, 0.5)]), cmap=cmz_map)
    plt.close(fig)

    ## Output
    t_start = 0
    t = 0
    fig, ax = plt.subplots(2, 2)
    plt.close(fig)

    plt.rcParams["figure.figsize"] = (17, 8)

    fig = plt.figure(figsize=(20, 10))

    gs0 = fig.add_gridspec(2, 1, height_ratios=[0.75, 1])
    gs1 = gs0[0].subgridspec(2, 2, height_ratios=[1, 0.3])
    gs2 = gs0[1].subgridspec(3, 2, hspace=1, height_ratios=[1, 1, 0.25])
    gs200 = gs2[0, 0].subgridspec(1, 2, width_ratios=[1, 0.75])
    gs210 = gs2[1, 0].subgridspec(1, 2, width_ratios=[1, 0.75])
    gs201 = gs2[0, 1].subgridspec(1, 2, width_ratios=[1, 0.75])
    gs211 = gs2[1, 1].subgridspec(1, 2, width_ratios=[1, 0.75])
    gscb = gs2[2, 1].subgridspec(1, 2, width_ratios=[1, 0.75])

    ax[0, 0] = fig.add_subplot(gs1[0, 0])
    [
        ax[0, 0].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)
        for i, ot in enumerate(Ot)
    ]
    ax[0, 0].spines[["right", "top"]].set_visible(False)
    ax[0, 0].set_xticks((0, 0.3, 0.6))
    ax[0, 0].set_yticks((0, 0.5, 1))
    ax[0, 0].set_ylim((0, 1.0))

    ax[0, 0].set_xlabel("time (s)")
    ax[0, 0].set_ylabel(("activation"))
    ax[0, 0].set_xlim((0, 0.6))
    ax[0, 0].set_title("dynamics with refraction")

    ax[1, 0] = fig.add_subplot(gs1[1, 1])
    ax[1, 0].axis("off")

    ax[0, 1] = fig.add_subplot(gs1[0, 1])
    l = [None] * 5
    for i, ot in enumerate(Ota):
        (l[i],) = ax[0, 1].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)

    ax[0, 1].spines[["top"]].set_visible(False)
    ax[0, 1].set_xticks((0, 0.3, 0.6))
    ax[0, 1].set_yticks((0, 0.5, 1))
    ax[0, 1].set_xlabel("time (s)")
    ax[0, 1].set_xlim((0, 0.6))
    ax[0, 1].set_ylim((0, 1.0))

    ax[0, 1].set_title("dynamics with refraction & alpha")

    ax2 = ax[0, 1].twinx()
    # ax2.plot(timevec,Rt[t_start+1:],color=lplot_c[1])
    (l[3],) = ax2.plot(
        timevec[t:],
        alpha_params[1][0]
        * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
        + alpha_params[1][0],
        color="k",
        linestyle="-.",
        linewidth=1.5,
    )
    (l[4],) = ax2.plot(
        timevec[t:],
        alpha_params[1][1]
        * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
        + alpha_params[1][1],
        color=np.array((0.5, 0.5, 0.5)),
        linestyle="-.",
        linewidth=1.5,
    )

    ax2.set_ylim((0, 1.0))
    ax2.set_yticks((0, 0.5, 1))
    ax2.set_ylabel("amplitude")

    ax2.spines[["top"]].set_visible(False)
    ax[1, 1] = fig.add_subplot(gs1[1, 1])
    ax[1, 1].axis("off")
    ax[1, 1].legend(
        l,
        ["A(t)", "E(t)", "T(t)", "alpha L1", "alpha L2"],
        loc="lower left",
        frameon=False,
        ncol=5,
    )

    ## Layer 1

    ## plot activations in nodes
    subax10 = fig.add_subplot(gs200[0])
    subax11 = fig.add_subplot(gs200[1])

    ## only active nodes
    Z = Z21.detach().cpu()
    Z_round = np.round(Z)
    Z_round_idx = np.argsort(Z_round)
    Z_round_sort = np.sort(Z_round)
    Z_round_unique_idx = np.unique(Z_round_sort, return_index=True)

    Z_idx = tuple(
        (
            Z_round_unique_idx[0][Z_round_unique_idx[0] > 0],
            Z_round_unique_idx[1][Z_round_unique_idx[0] > 0],
        )
    )

    for i, z in enumerate(Z_idx[1]):
        subax10.plot(timevec[t:], H1t[Z_round_idx[z], t + 1 :], color=cmz[i])

        subax11.plot(R1t[Z_round_idx[z], :], H1t[Z_round_idx[z], :], color=cmz[i])

        for i, z in enumerate(Z_idx[1][:-1]):
            plt.plot(
                Rstar1[Z_round_idx[z]],
                Hstar1[Z_round_idx[z]],
                "d",
                color=cmz[i],
                mec="k",
            )

    subax10.spines[["top", "right"]].set_visible(False)
    subax11.spines[["top", "right"]].set_visible(False)

    ## plot activations in nodes
    subax12 = fig.add_subplot(gs201[0])
    subax13 = fig.add_subplot(gs201[1])

    ## only active nodes
    Z = Z21.detach().cpu()
    Z_round = np.round(Z)
    Z_round_idx = np.argsort(Z_round)
    Z_round_sort = np.sort(Z_round)
    Z_round_unique_idx = np.unique(Z_round_sort, return_index=True)

    Z_idx = tuple(
        (
            Z_round_unique_idx[0][Z_round_unique_idx[0] > 0],
            Z_round_unique_idx[1][Z_round_unique_idx[0] > 0],
        )
    )

    for i, z in enumerate(Z_idx[1]):
        subax12.plot(timevec[t:], H1ta[Z_round_idx[z], t + 1 :], color=cmz[i])
        subax13.plot(R1ta[Z_round_idx[z], :], H1ta[Z_round_idx[z], :], color=cmz[i])
        for i, z in enumerate(Z_idx[1]):
            subax13.plot(
                Rstar1[Z_round_idx[z]],
                Hstar1[Z_round_idx[z]],
                "d",
                color=cmz[i],
                mec="k",
            )

    subax12.spines[["top", "right"]].set_visible(False)
    subax13.spines[["top", "right"]].set_visible(False)

    max_R = torch.ceil(torch.max(R1t))
    subax13.set_xlim((0, max_R))
    subax13.set_xticks(np.arange(0, max_R+1, max_R/2))
    subax11.set_xlim((0, max_R))
    subax11.set_xticks(np.arange(0, max_R+1, max_R/2))

    ## Layer 2

    ## plot activations in nodes
    subax20 = fig.add_subplot(gs210[0])
    subax21 = fig.add_subplot(gs210[1])

    ## only active nodes
    Z = Z22.detach().cpu()
    Z_round = np.round(Z)
    Z_round_idx = np.argsort(Z_round)
    Z_round_sort = np.sort(Z_round)
    Z_round_unique_idx = np.unique(Z_round_sort, return_index=True)

    Z_idx = tuple(
        (
            Z_round_unique_idx[0][Z_round_unique_idx[0] > 0],
            Z_round_unique_idx[1][Z_round_unique_idx[0] > 0],
        )
    )

    for i, z in enumerate(Z_idx[1]):
        subax20.plot(timevec[t:], H2t[Z_round_idx[z], t + 1 :], color=cmz[i])
        subax21.plot(R2t[Z_round_idx[z], :], H2t[Z_round_idx[z], :], color=cmz[i])

    for i, z in enumerate(Z_idx[1]):
        plt.plot(
            Rstar2[Z_round_idx[z]], Hstar2[Z_round_idx[z]], "d", color=cmz[i], mec="k"
        )

    subax20.spines[["top", "right"]].set_visible(False)
    subax21.spines[["top", "right"]].set_visible(False)
    subax20.set_xlim((0, 0.6))
    subax20.set_xticks(np.arange(0, 0.7, 0.3))

    ## plot activations in nodes
    subax22 = fig.add_subplot(gs211[0])
    subax23 = fig.add_subplot(gs211[1])

    ## only active nodes
    Z = Z22.detach().cpu()
    Z_round = np.round(Z)
    Z_round_idx = np.argsort(Z_round)
    Z_round_sort = np.sort(Z_round)
    Z_round_unique_idx = np.unique(Z_round_sort, return_index=True)

    Z_idx = tuple(
        (
            Z_round_unique_idx[0][Z_round_unique_idx[0] > 0],
            Z_round_unique_idx[1][Z_round_unique_idx[0] > 0],
        )
    )

    for i, z in enumerate(Z_idx[1]):
        subax22.plot(timevec[t:], H2ta[Z_round_idx[z], t + 1 :], color=cmz[i])
        subax23.plot(R2ta[Z_round_idx[z], :], H2ta[Z_round_idx[z], :], color=cmz[i])
        for i, z in enumerate(Z_idx[1]):
            plt.plot(
                Rstar2[Z_round_idx[z]],
                Hstar2[Z_round_idx[z]],
                "d",
                color=cmz[i],
                mec="k",
            )

    subax23.set_xlim((0, max_R))
    subax23.set_xticks(np.arange(0, max_R+1, max_R/2))
    subax21.set_xlim((0, max_R))
    subax21.set_xticks(np.arange(0, max_R+1, max_R/2))

    subax22.spines[["top", "right"]].set_visible(False)
    subax23.spines[["top", "right"]].set_visible(False)
    subax10.set_xlim((0, 0.6))
    subax10.set_xticks(np.arange(0, 0.7, 0.3))

    subax12.set_xlim((0, 0.6))
    subax12.set_xticks(np.arange(0, 0.7, 0.3))

    subax20.set_xlim((0, 0.6))
    subax20.set_xticks(np.arange(0, 0.7, 0.3))

    subax22.set_xlim((0, 0.6))
    subax22.set_xticks(np.arange(0, 0.7, 0.3))

    subax10.set_ylim((0, 1))
    subax20.set_ylim((0, 1))
    subax11.set_ylim((0, 1))
    subax21.set_ylim((0, 1))
    subax12.set_ylim((0, 1))
    subax22.set_ylim((0, 1))

    subax13.set_ylim((0, 1))
    subax23.set_ylim((0, 1))

    subax10.set_yticks((0, 0.5, 1))
    subax20.set_yticks((0, 0.5, 1))
    subax11.set_yticks(())
    subax21.set_yticks(())
    subax12.set_yticks((0, 0.5, 1))
    subax22.set_yticks((0, 0.5, 1))

    subax13.set_yticks(())
    subax23.set_yticks(())

    subax11.set_yticklabels(())
    subax21.set_yticklabels(())

    subax13.set_yticklabels(())
    subax23.set_yticklabels(())

    subax10.set_ylabel("H")
    subax20.set_ylabel("H")
    subax10.set_xlabel("time(s)")
    subax20.set_xlabel("time(s)")

    subax12.set_ylabel("H")
    subax22.set_ylabel("H")
    subax12.set_xlabel("time(s)")
    subax22.set_xlabel("time(s)")

    subax11.set_xlabel("R")
    subax21.set_xlabel("R")

    subax13.set_xlabel("R")
    subax23.set_xlabel("R")

    # titles
    subax20.set_title("Layer 2", loc="left", pad=10)
    subax22.set_title("Layer 2", loc="left", pad=10)
    subax10.set_title("Layer 1", loc="left", pad=10)
    subax12.set_title("Layer 1", loc="left", pad=10)

    subcb = fig.add_subplot(gscb[1])
    subcb.axis("off")
    cb = fig.colorbar(imfig, ax=subcb, orientation="horizontal", fraction=1)
    cb.set_label("Z")
    cb.set_ticks(np.arange(0, maxZ + 1, (maxZ + 0.5) / 2))

    return fig


def fig5(x_train, model2, idx, label):

    # parameters dynamics
    tau_h = 0.01
    tau_R = 0.1
    S = 1
    c = 0

    t_start = 0  # when stimulus presented
    t = 0  # when to begin plot

    params = [tau_h, tau_R, c, S]

    # alpha parameters
    afreq = [10, 10]
    aamp = [0, 0]
    aph = [np.pi / 2 + 2 * np.pi / 100 * 10, np.pi / 2]

    alpha_params = [afreq, aamp, aph]

    # combination of stimuli

    inp_combi = list(combinations(idx, 2))  # possible combinations

    timevec = np.linspace(0, 1, 1000)

    plt.rcParams["figure.figsize"] = (6, 6)

    fig, axs = plt.subplots(4, 2, height_ratios=[1, 1, 1, 0.5], width_ratios=[0.5, 1])

    inpt = [None] * 3
    for i, comp_inp in enumerate(inp_combi):

        inpt[i] = x_train[comp_inp[0]] + x_train[comp_inp[1]]

        _, _, _, Ocmb = model2.forw_conv(inpt[i])

        # imshow
        imfig = axs[i, 0].imshow(
            Ocmb.reshape(-1, 1).detach().cpu().numpy(),
            cm.lajolla_r,
            interpolation="nearest",
            aspect=1,
        )
        imfig.set_clim(0, 1)
        axs[i, 0].set_yticks(np.arange(0, 3, 1))
        axs[i, 0].set_yticklabels(label)

        axs[i, 0].set_xticks(())

        axs[i, 0].spines[:].set_visible(False)

        axs[i, 0].set_yticks(np.arange(3))

        axs[i, 0].set_yticklabels(["A", "E", "T"])

        axs[i, 0].set_yticks(np.arange(-0.5, 3, 1), minor=True)
        axs[i, 0].grid(which="minor", color="w", linestyle="-", linewidth=3)
        axs[i, 0].tick_params(which="minor", bottom=False, left=False)

        # time course
        Z1t, Z2t, H1t, R1t, H2t, R2t, Ot = euler_dyn_2layer(
            model2,
            inpt[i],
            params,
            timevec,
            alpha_params,
            DEVICE,
            inp_on=t_start,
            start_fix=False,
        )

        l = [None] * 3
        for io, o in enumerate(Ot):
            (l[io],) = axs[i, 1].plot(
                timevec, o[t + 1 :], linewidth=3, color=lpcm[io], label=label[io]
            )
        axs[i, 1].set_ylim((0, 1))
        axs[i, 1].set_xlim((0, 0.3))
        axs[i, 1].set_xticks(np.arange(0, 0.31, 0.1))

        axs[i, 1].spines[["top", "right"]].set_visible(False)
        axs[i, 1].set_ylabel("activation")
        axs[i, 1].set_ylim(0, 1.1)
        axs[i, 1].set_xlabel("time (s)")

        axs[3, 0].axis("off")
        axs[3, 1].axis("off")
        axs[3, 1].legend(l, label, loc="lower right", ncol=3, frameon=False)

    cb = fig.colorbar(imfig, ax=axs[3, 0], orientation="horizontal", fraction=0.5)
    cb.set_label("activation")
    cb.set_ticks((0, 0.5, 1))

    fig.tight_layout()

    return fig, inpt


def fig6(timevec, t, alpha_params, OtEA, OtaEA, OtET, OtaET, OtAT, OtaAT):

    fig, ax = plt.subplots(4, 2)
    plt.close(fig)  # This will prevent the figure from being displayed

    # plot

    fig = plt.figure(figsize=(15, 9))

    # E & A
    gs0 = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.3])

    ax[0, 0] = fig.add_subplot(gs0[0, 0])
    [
        ax[0, 0].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)
        for i, ot in enumerate(OtEA)
    ]
    ax[0, 0].spines[["right", "top"]].set_visible(False)
    ax[0, 0].set_xticks((0, 0.3, 0.6))
    ax[0, 0].set_yticks((0, 0.5, 1))
    ax[0, 0].set_xlabel("time (s)")
    ax[0, 0].set_ylabel(("activation"))
    ax[0, 0].set_xlim((0, 0.6))
    ax[0, 0].set_title("dynamics with refraction")

    ax[0, 1] = fig.add_subplot(gs0[0, 1])
    l = [None] * 5
    for i, ot in enumerate(OtaEA):
        (l[i],) = ax[0, 1].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)

    ax[0, 1].spines[["right", "top"]].set_visible(False)
    ax[0, 1].set_xticks((0, 0.3, 0.6))
    ax[0, 1].set_yticks((0, 0.5, 1))
    ax[0, 1].set_xlabel("time (s)")
    ax[0, 1].set_xlim((0, 0.6))
    ax[0, 1].set_title("dynamics with refraction & alpha")

    ax2 = ax[0, 1].twinx()
    (l[3],) = ax2.plot(
        timevec[t:],
        alpha_params[1][0]
        * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
        + alpha_params[1][0],
        color="k",
        linestyle="-.",
        linewidth=1.5,
    )
    (l[4],) = ax2.plot(
        timevec[t:],
        alpha_params[1][1]
        * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
        + alpha_params[1][1],
        color=np.array((0.5, 0.5, 0.5)),
        linestyle="-.",
        linewidth=1.5,
    )

    ax2.set_ylim((0, 1.0))
    ax2.set_yticks((0, 0.5, 1))
    ax2.set_ylabel("amplitude")

    ax2.spines[["top"]].set_visible(False)

    # E & T

    ax[1, 0] = fig.add_subplot(gs0[1, 0])
    [
        ax[1, 0].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)
        for i, ot in enumerate(OtET)
    ]
    ax[1, 0].spines[["right", "top"]].set_visible(False)
    ax[1, 0].set_xticks((0, 0.3, 0.6))

    ax[1, 0].set_yticks((0, 0.5, 1))
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].set_ylabel(("activation"))
    ax[1, 0].set_xlim((0, 0.6))

    ax[1, 1] = fig.add_subplot(gs0[1, 1])
    l = [None] * 5
    for i, ot in enumerate(OtaET):
        (l[i],) = ax[1, 1].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)

    ax[1, 1].spines[["right", "top"]].set_visible(False)
    ax[1, 1].set_xticks((0, 0.3, 0.6))
    ax[1, 1].set_yticks((0, 0.5, 1))
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_xlim((0, 0.6))

    ax2 = ax[1, 1].twinx()
    (l[3],) = ax2.plot(
        timevec[t:],
        alpha_params[1][0]
        * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
        + alpha_params[1][0],
        color="k",
        linestyle="-.",
        linewidth=1.5,
    )
    (l[4],) = ax2.plot(
        timevec[t:],
        alpha_params[1][1]
        * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
        + alpha_params[1][1],
        color=np.array((0.5, 0.5, 0.5)),
        linestyle="-.",
        linewidth=1.5,
    )

    ax2.set_ylim((0, 1.0))
    ax2.set_yticks((0, 0.5, 1))
    ax2.set_ylabel("amplitude")

    ax2.spines[["top"]].set_visible(False)

    ax[0, 0].set_ylim((0, 1))
    ax[0, 1].set_ylim((0, 1))
    ax[1, 0].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))

    ax[1, 1] = fig.add_subplot(gs0[1, 1])
    l = [None] * 5
    for i, ot in enumerate(OtaET):
        (l[i],) = ax[1, 1].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)

    ax[1, 1].spines[["right", "top"]].set_visible(False)
    ax[1, 1].set_xticks((0, 0.3, 0.6))
    ax[1, 1].set_yticks((0, 0.5, 1))
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_xlim((0, 0.6))

    ax2 = ax[1, 1].twinx()
    (l[3],) = ax2.plot(
        timevec[t:],
        alpha_params[1][0]
        * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
        + alpha_params[1][0],
        color="k",
        linestyle="-.",
        linewidth=1.5,
    )
    (l[4],) = ax2.plot(
        timevec[t:],
        alpha_params[1][1]
        * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
        + alpha_params[1][1],
        color=np.array((0.5, 0.5, 0.5)),
        linestyle="-.",
        linewidth=1.5,
    )

    ax2.set_ylim((0, 1.0))
    ax2.set_yticks((0, 0.5, 1))
    ax2.set_ylabel("amplitude")

    ax2.spines[["top"]].set_visible(False)

    ax[0, 0].set_ylim((0, 1))
    ax[0, 1].set_ylim((0, 1))
    ax[1, 0].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))

    # A & T

    ax[2, 0] = fig.add_subplot(gs0[2, 0])
    [
        ax[2, 0].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)
        for i, ot in enumerate(OtAT)
    ]
    ax[2, 0].spines[["right", "top"]].set_visible(False)
    ax[2, 0].set_xticks((0, 0.3, 0.6))

    ax[2, 0].set_yticks((0, 0.5, 1))
    ax[2, 0].set_xlabel("time (s)")
    ax[2, 0].set_ylabel(("activation"))
    ax[2, 0].set_xlim((0, 0.6))

    ax[2, 1] = fig.add_subplot(gs0[2, 1])
    l = [None] * 5
    for i, ot in enumerate(OtaAT):
        (l[i],) = ax[2, 1].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)

    ax[2, 1].spines[["right", "top"]].set_visible(False)
    ax[2, 1].set_xticks((0, 0.3, 0.6))
    ax[2, 1].set_yticks((0, 0.5, 1))
    ax[2, 1].set_xlabel("time (s)")
    ax[2, 1].set_xlim((0, 0.6))

    ax2 = ax[2, 1].twinx()
    (l[3],) = ax2.plot(
        timevec[t:],
        alpha_params[1][0]
        * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
        + alpha_params[1][0],
        color="k",
        linestyle="-.",
        linewidth=1.5,
    )
    (l[4],) = ax2.plot(
        timevec[t:],
        alpha_params[1][1]
        * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
        + alpha_params[1][1],
        color=np.array((0.5, 0.5, 0.5)),
        linestyle="-.",
        linewidth=1.5,
    )

    ax2.set_ylim((0, 1.0))
    ax2.set_yticks((0, 0.5, 1))
    ax2.set_ylabel("amplitude")

    ax2.spines[["top"]].set_visible(False)

    ax[0, 0].set_ylim((0, 1))
    ax[0, 1].set_ylim((0, 1))
    ax[1, 0].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[2, 0].set_ylim((0, 1))
    ax[2, 1].set_ylim((0, 1))

    ax[3, 0] = fig.add_subplot(gs0[3, 0])
    ax[3, 0].axis("off")
    ax[3, 1] = fig.add_subplot(gs0[3, 1])
    ax[3, 1].axis("off")
    ax[3, 1].legend(
        l,
        ["A(t)", "E(t)", "T(t)", "alpha L1", "alpha L2"],
        loc="lower left",
        frameon=False,
        ncol=5,
    )

    fig.tight_layout()

    return fig


def fig7(
    timevec,
    t,
    alpha_params,
    H1corrA,
    H1corrE,
    H1corrT,
    H1t,
    H2corrA,
    H2corrE,
    H2corrT,
    H2t,
    Ot,
):

    plt.rcParams["figure.figsize"] = (17, 6)
    matplotlib.rcParams.update({"font.size": 14})
    # plots
    # 1 temporal code
    # 2 similarity H1 to activation each stimulus alone
    # 3 TFR style activation H1 over time

    # prepare layout
    fig, axs = plt.subplots(7, 1)  # empty
    plt.close(fig)
    fig, axs2 = plt.subplots(7, 1)  # empty
    plt.close(fig)

    fig = plt.figure(figsize=(17, 8))

    gs = GridSpec(1, 3, figure=fig)

    gs01 = gs[0].subgridspec(3, 1, height_ratios=[0.4, 1, 0.05])
    gs02 = gs[1].subgridspec(3, 1, height_ratios=[0.4, 1, 0.05])
    gs03 = gs[2].subgridspec(3, 1, height_ratios=[0.4, 1, 0.05])

    axs[0] = fig.add_subplot(gs01[0])
    axs[1] = fig.add_subplot(gs01[1])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    axs[2] = fig.add_subplot(gs02[0])
    axs[3] = fig.add_subplot(gs02[1])
    axs[4] = fig.add_subplot(gs03[0])
    axs[5] = fig.add_subplot(gs03[1])
    axs[6] = fig.add_subplot(gs03[2])

    axs[0].plot(timevec, H1corrA[t + 1 :], color=lpcm[0], linewidth=3)
    axs[0].plot(timevec, H1corrE[t + 1 :], color=lpcm[1], linewidth=3)
    axs[0].plot(timevec, H1corrT[t + 1 :], color=lpcm[2], linewidth=3)

    axs[0].set_ylim([0, 1.1])
    axs[0].set_yticks(np.arange(0, 1.1, 0.5))
    axs[0].set_xlim([0, 1])
    axs[0].spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("similarity (norm. dot)")

    iml1 = axs[1].imshow(H1t, aspect=5, cmap=cm.lajolla_r)
    iml1.set_clim(0, 1)
    axs[1].set_xticks(np.arange(0, 1200, 200))
    axs[1].set_xticklabels(np.round(np.arange(0, 1.2, 0.2), decimals=1))
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylim([0, 64])
    axs[1].set_ylabel("neuron index")

    # Layer 2

    axs[2].plot(timevec, H2corrA[t + 1 :], color=lpcm[0], linewidth=3)
    axs[2].plot(timevec, H2corrE[t + 1 :], color=lpcm[1], linewidth=3)
    axs[2].plot(timevec, H2corrT[t + 1 :], color=lpcm[2], linewidth=3)

    axs[2].set_ylim([0, 1.1])
    axs[2].set_yticks(())
    axs[2].set_xlim([0, 1])
    axs[2].spines[["top", "right"]].set_visible(False)

    iml2 = axs[3].imshow(H2t, aspect=10, cmap=cm.lajolla_r)
    iml2.set_clim(0, 1)
    axs[3].set_xticks(np.arange(0, 1200, 200))
    axs[3].set_xticklabels(np.round(np.arange(0, 1.2, 0.2), decimals=1))
    axs[3].set_xlabel("time (s)")
    axs[3].set_yticks(np.arange(0, 33, 6))
    axs[3].set_yticklabels(np.arange(1, 32, 6))
    axs[3].set_ylim([0, 31])

    # Output layer

    [
        axs[4].plot(timevec, ot[1:], linewidth=3, color=lpcm[iot])
        for iot, ot in enumerate(Ot)
    ]

    axs[4].set_ylim([0, 1.1])
    axs[4].set_yticks((0, 0.5, 1))
    axs[4].set_ylabel("activation")

    axs[4].spines[["top", "right"]].set_visible(False)

    imout = axs[5].imshow(Ot, aspect=20, cmap=cm.lajolla_r)
    axs[5].set_xticks(np.arange(0, 1200, 200))
    axs[5].set_xticklabels(np.round(np.arange(0, 1.2, 0.2), decimals=1))
    axs[5].set_xlabel("time (s)")
    axs[5].set_yticks(np.arange(0, 3, 1))
    axs[5].set_yticklabels(["A", "E", "T"])

    # clean up axes
    axs[0].set_xlim((0, 0.4))
    axs[1].set_xlim((0, 400))
    axs[2].set_xlim((0, 0.4))
    axs[3].set_xlim((0, 400))
    axs[4].set_xlim((0, 0.4))
    axs[5].set_xlim((0, 400))

    axs[0].set_xticks(np.arange(0, 0.6, 0.2))
    axs[2].set_xticks(np.arange(0, 0.6, 0.2))
    axs[4].set_xticks(np.arange(0, 0.6, 0.2))

    # add alpha inhibition plot to the line graphs
    axs2[0] = axs[0].twinx()
    axs2[0].plot(
        timevec[t:],
        alpha_params[1][0]
        * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
        + alpha_params[1][0],
        color="k",
        linestyle="-.",
    )
    axs2[0].set_ylim((0, 1.1))
    axs2[0].set_yticks(())
    axs2[0].spines[["top"]].set_visible(False)

    axs2[2] = axs[2].twinx()
    axs2[2].plot(
        timevec[t:],
        alpha_params[1][1]
        * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
        + alpha_params[1][1],
        color=np.array((0.5, 0.5, 0.5)),
        linestyle="-.",
    )
    axs2[2].set_ylim((0, 1.1))
    axs2[2].set_yticks(())
    axs2[2].spines[["top"]].set_visible(False)

    axs2[4] = axs[4].twinx()
    axs2[4].set_ylim((0, 1.1))
    axs2[4].set_yticks((0, 0.5, 1))

    axs2[4].set_ylabel("amplitude")
    axs2[4].spines[["top"]].set_visible(False)

    axs2[4].plot(
        timevec[t:],
        alpha_params[1][0]
        * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
        + alpha_params[1][0],
        color="k",
        linestyle="-.",
    )
    axs2[4].plot(
        timevec[t:],
        alpha_params[1][1]
        * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
        + alpha_params[1][1],
        color=np.array((0.5, 0.5, 0.5)),
        linestyle="-.",
    )

    imout.set_clim((0, 1))

    axs[6].axis("off")
    cb = fig.colorbar(imout, ax=axs[6], orientation="horizontal", fraction=0.75)
    cb.set_label("activation")
    cb.set_ticks((0, 0.5, 1))

    fig.tight_layout()

    return fig


def fig8(timevec, t_start, t, inpt, model2, params, alpha_params, ph_diff, start_fix_z):

    fig, axs = plt.subplots(4, 3, figsize=[21, 9])
    ax2 = axs.copy()
    axs = axs.ravel()
    ax2 = ax2.ravel()
    afreq, aamp = alpha_params

    for i, ph in enumerate(ph_diff):
        aph = [np.pi / 2, np.pi / 2 - ph]
        alpha_params = [afreq, aamp, aph]

        Ot = euler_dyn_2layer(
            model2,
            inpt,
            params,
            timevec,
            alpha_params,
            DEVICE,
            inp_on=t_start,
            start_fix=start_fix_z,
        )[-1]

        [
            axs[i].plot(timevec, ot[1:], linewidth=2, color=lpcm[iot])
            for iot, ot in enumerate(Ot)
        ]

        axs[i].set_title(str(np.round(100 / (2 * np.pi) * ph, decimals=1)) + " ms")

        axs[i].set_xlim((0, 0.6))
        axs[i].spines[["top"]].set_visible(False)
        axs[i].set_ylim((0, 1.0))

        ax2[i] = axs[i].twinx()
        ax2[i].plot(
            timevec[t:],
            alpha_params[1][0]
            * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
            + alpha_params[1][0],
            color="k",
            linestyle="-.",
            linewidth=1.5,
        )
        ax2[i].plot(
            timevec[t:],
            alpha_params[1][1]
            * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
            + alpha_params[1][1],
            color=np.array((0.5, 0.5, 0.5)),
            linestyle="-.",
            linewidth=1.5,
        )

        ax2[i].set_ylim((0, 1.0))

        if i == 0 or i == 3 or i == 6 or i == 9:
            axs[i].set_yticks((0, 0.5, 1))
            axs[i].set_ylabel("softm. activation")
        else:
            axs[i].set_yticks(())

        if i == 2 or i == 5 or i == 8 or i == 11:
            ax2[i].set_yticks((0, 0.5, 1))
            ax2[i].set_ylabel("a amplitude")
        else:
            ax2[i].set_yticks(())

        if i > 8:
            axs[i].set_xticks((0, 0.3, 0.6))
            axs[i].set_xlabel("time (s)")
        else:
            axs[i].set_xticks(())

        ax2[i].spines[["top"]].set_visible(False)

        axs[i].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    return fig


def supp_fig1(
    x_train,
    model2,
    timevec,
    t_start,
    t,
    params,
    alpha_params,
    idx,
    labels,
    start_fix_z,
    noise=0,
):

    inp_combi = list(combinations(idx, 2))  # possible combinations

    label_comb = list(combinations(labels, 2))
    label_comb

    plt.rcParams["figure.figsize"] = (17, 5)

    # prep second axis
    fig, ax2 = plt.subplots(2, 3)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3)

    # store accuracy
    qu_acc = np.zeros((len(inp_combi)*2,2))

    # use this array to find the correct output node correspnding to input node
    stim_rang = np.arange(0,12).reshape(3,4)

    for i, comp_inp in enumerate(inp_combi):

        inpt = (
            x_train[comp_inp[0]] * 1.2 + x_train[comp_inp[1]] * 0.8 + torch.normal(0.4, 0.1, x_train[comp_inp[0]].shape) * noise
        )
        inpt /= torch.max(inpt)

        Z1t, Z2t, H1t, R1t, H2t, R2t, Ot = euler_dyn_2layer(
            model2,
            inpt,
            params,
            timevec,
            alpha_params,
            DEVICE,
            inp_on=t_start,
            start_fix=start_fix_z,
        )

        # quantify read-out accuracy
        node0 = np.where(stim_rang==comp_inp[0])[0]
        node1 = np.where(stim_rang==comp_inp[1])[0]

        acc0 = np.round(torch.max(Ot[node0]).numpy(),decimals=2)
        acc1 = np.round(torch.max(Ot[node1]).numpy(),decimals=2)
        
        qu_acc[i] = np.array((acc0, acc1))


        [
            axs[0, i].plot(timevec, ot[1:], linewidth=2, color=lpcm[iot])
            for iot, ot in enumerate(Ot)
        ]

        axs[0, i].set_title(label_comb[i][0] + " + " + label_comb[i][1])

        axs[0, i].set_xlim((0, 0.8))
        axs[0, i].set_xticks(())
        axs[0, i].spines[["top"]].set_visible(False)
        axs[0, i].set_ylim((0, 1.0))
        ax2[0, i] = axs[0, i].twinx()
        ax2[0, i].set_ylim((0, 1.0))
        ax2[0, i].set_yticks((0, 0.5, 1))

        ax2[0, i].plot(
            timevec[t:],
            alpha_params[1][0]
            * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
            + alpha_params[1][0],
            color="k",
            linestyle="-.",
            linewidth=1.5,
        )
        ax2[0, i].plot(
            timevec[t:],
            alpha_params[1][1]
            * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
            + alpha_params[1][1],
            color=np.array((0.5, 0.5, 0.5)),
            linestyle="-.",
            linewidth=1.5,
        )

        ax2[0, i].spines[["top"]].set_visible(False)
        ax2[0, i].set_yticks(())


        loc0 = timevec[np.argmax(Ot[node0][:800].numpy())]
        loc1 = timevec[np.argmax(Ot[node1][:800].numpy())]

        axs[0,i].text(loc0,acc0+.07,str(acc0),color=lpcm[node0])
        axs[0,i].text(loc1,acc1+.04,str(acc1),color=lpcm[node1])

    # clean up axes
    axs[0, 0].set_ylabel("activation")
    axs[0, 0].set_yticks((0, 0.5, 1))
    ax2[0, 2].set_ylabel("amplitude")
    ax2[0, 2].set_yticks((0, 0.5, 1))

    for i, comp_inp in enumerate(inp_combi):

        inpt = (
            x_train[comp_inp[1]] * 1.2 + x_train[comp_inp[0]] * 0.8 + torch.normal(0.4, 0.1, x_train[comp_inp[0]].shape) * noise
        )
        inpt /= torch.max(inpt)

        Z1t, Z2t, H1t, R1t, H2t, R2t, Ot = euler_dyn_2layer(
            model2,
            inpt,
            params,
            timevec,
            alpha_params,
            DEVICE,
            inp_on=t_start,
            start_fix=True,
        )

        [
            axs[1, i].plot(timevec, ot[1:], linewidth=2, color=lpcm[iot])
            for iot, ot in enumerate(Ot)
        ]

         # quantify read-out accuracy
        node0 = np.where(stim_rang==comp_inp[1])[0]
        node1 = np.where(stim_rang==comp_inp[0])[0]
        loc0 = timevec[np.argmax(Ot[node0][:800].numpy())]
        loc1 = timevec[np.argmax(Ot[node1][:800].numpy())]

        acc0 = np.round(torch.max(Ot[node0]).numpy(),decimals=2)
        acc1 = np.round(torch.max(Ot[node1]).numpy(),decimals=2)

        qu_acc[i+2] = np.array((acc0, acc1))

        axs[1,i].text(loc0,acc0+.07,str(acc0),color=lpcm[node0])
        axs[1,i].text(loc1,acc1+.04,str(acc1),color=lpcm[node1])

        axs[1, i].set_title(label_comb[i][1] + " + " + label_comb[i][0])

        axs[1, i].set_xlabel("time (s)")
        axs[1, i].spines[["top"]].set_visible(False)

        ax2[1, i] = axs[1, i].twinx()
        ax2[1, i].plot(
            timevec[t:],
            alpha_params[1][0]
            * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
            + alpha_params[1][0],
            color="k",
            linestyle="-.",
            linewidth=1.5,
        )
        ax2[1, i].plot(
            timevec[t:],
            alpha_params[1][1]
            * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
            + alpha_params[1][1],
            color=np.array((0.5, 0.5, 0.5)),
            linestyle="-.",
            linewidth=1.5,
        )

        ax2[1, i].set_ylim((0, 1.0))
        ax2[1, i].set_yticks(())

        axs[1, i].set_ylim((0, 1.0))
        axs[1, i].set_xlim((0, 0.8))
        axs[1, i].set_xticks((0, 0.4, 0.8))

        ax2[1, i].spines[["top"]].set_visible(False)

    axs[1, 1].set_yticks(())
    axs[1, 2].set_yticks(())

    axs[0, 1].set_yticks(())
    axs[0, 2].set_yticks(())

    ax2[1, 2].set_yticks((0, 0.5, 1))
    ax2[1, 2].set_ylabel("amplitude")

    # clean up axes
    axs[1, 0].set_ylabel("activation")
    axs[1, 0].set_yticks((0, 0.5, 1))

    return fig, qu_acc


def supp_fig4(
    timevec,
    t,
    alpha_params_old_dyn,
    alpha_params_new_dyn,
    Ot_old_dyn,
    Ota_old_dyn,
    Ot_new_dyn,
    Ota_new_dyn,
):

    fig, ax = plt.subplots(4, 2)
    plt.close(fig)  # This will prevent the figure from being displayed

    # plot

    fig = plt.figure(figsize=(15, 6))

    # old dynamics (tau_h=0.01, tau_r = 0.1)
    gs0 = fig.add_gridspec(2, 2)

    ax[0, 0] = fig.add_subplot(gs0[0, 0])
    [
        ax[0, 0].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)
        for i, ot in enumerate(Ot_old_dyn)
    ]
    ax[0, 0].spines[["right", "top"]].set_visible(False)
    ax[0, 0].set_xticks((0, 0.4, 0.8))
    ax[0, 0].set_yticks((0, 0.5, 1))
    ax[0, 0].set_xlabel("time (s)")
    ax[0, 0].set_ylabel(("activation"))
    ax[0, 0].set_xlim((0, 0.8))
    ax[0, 0].set_title("dynamics with refraction")

    ax[0, 1] = fig.add_subplot(gs0[0, 1])
    l = [None] * 5
    for i, ot in enumerate(Ota_old_dyn):
        (l[i],) = ax[0, 1].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)

    ax[0, 1].spines[["right", "top"]].set_visible(False)
    ax[0, 1].set_xticks((0, 0.4, 0.8))
    ax[0, 1].set_yticks((0, 0.5, 1))
    ax[0, 1].set_xlabel("time (s)")
    ax[0, 1].set_xlim((0, 0.8))
    ax[0, 1].set_title("dynamics with refraction & alpha")

    ax2 = ax[0, 1].twinx()
    (l[3],) = ax2.plot(
        timevec[t:],
        alpha_params_old_dyn[1][0]
        * np.sin(
            2 * np.pi * alpha_params_old_dyn[0][0] * timevec[t:]
            + alpha_params_old_dyn[2][0]
        )
        + alpha_params_old_dyn[1][0],
        color="k",
        linestyle="-.",
        linewidth=1.5,
    )
    (l[4],) = ax2.plot(
        timevec[t:],
        alpha_params_old_dyn[1][1]
        * np.sin(
            2 * np.pi * alpha_params_old_dyn[0][1] * timevec[t:]
            + alpha_params_old_dyn[2][1]
        )
        + alpha_params_old_dyn[1][1],
        color=np.array((0.5, 0.5, 0.5)),
        linestyle="-.",
        linewidth=1.5,
    )

    ax2.set_ylim((0, 1.0))
    ax2.set_yticks((0, 0.5, 1))
    ax2.set_ylabel("amplitude")

    ax2.spines[["top"]].set_visible(False)

    # faster dynamics (tau_h=0.005, tau_r = 0.05)

    ax[1, 0] = fig.add_subplot(gs0[1, 0])
    [
        ax[1, 0].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)
        for i, ot in enumerate(Ot_new_dyn)
    ]
    ax[1, 0].spines[["right", "top"]].set_visible(False)
    ax[1, 0].set_xticks((0, 0.4, 0.8))

    ax[1, 0].set_yticks((0, 0.5, 1))
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].set_ylabel(("activation"))
    ax[1, 0].set_xlim((0, 0.8))

    ax[1, 1] = fig.add_subplot(gs0[1, 1])
    l = [None] * 5
    for i, ot in enumerate(Ota_new_dyn):
        (l[i],) = ax[1, 1].plot(timevec, ot[1:], color=lpcm[i], linewidth=3)

    ax[1, 1].spines[["right", "top"]].set_visible(False)
    ax[1, 1].set_xticks((0, 0.4, 0.8))
    ax[1, 1].set_yticks((0, 0.5, 1))
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_xlim((0, 0.8))

    ax2 = ax[1, 1].twinx()
    (l[3],) = ax2.plot(
        timevec[t:],
        alpha_params_new_dyn[1][0]
        * np.sin(
            2 * np.pi * alpha_params_new_dyn[0][0] * timevec[t:]
            + alpha_params_new_dyn[2][0]
        )
        + alpha_params_new_dyn[1][0],
        color="k",
        linestyle="-.",
        linewidth=1.5,
    )
    (l[4],) = ax2.plot(
        timevec[t:],
        alpha_params_new_dyn[1][1]
        * np.sin(
            2 * np.pi * alpha_params_new_dyn[0][1] * timevec[t:]
            + alpha_params_new_dyn[2][1]
        )
        + alpha_params_new_dyn[1][1],
        color=np.array((0.5, 0.5, 0.5)),
        linestyle="-.",
        linewidth=1.5,
    )

    ax2.set_ylim((0, 1.0))
    ax2.set_yticks((0, 0.5, 1))
    ax2.set_ylabel("amplitude")

    ax2.spines[["top"]].set_visible(False)

    ax[0, 0].set_ylim((0, 1))
    ax[0, 1].set_ylim((0, 1))
    ax[1, 0].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))

    fig.tight_layout()

    return fig

def supp_fig5(timevec, t_start, t, inpt, model2, params, alpha_params, ph_diff, start_fix_z):

    fig, axs = plt.subplots(4, 3, figsize=[21, 9])
    ax2 = axs.copy()
    axs = axs.ravel()
    ax2 = ax2.ravel()
    afreq, aamp = alpha_params

    for i, ph in enumerate(ph_diff):
        aph = [np.pi / 2 - ph, np.pi / 2]
        alpha_params = [afreq, aamp, aph]

        Ot = euler_dyn_2layer(
            model2,
            inpt,
            params,
            timevec,
            alpha_params,
            DEVICE,
            inp_on=t_start,
            start_fix=start_fix_z,
        )[-1]

        [
            axs[i].plot(timevec, ot[1:], linewidth=2, color=lpcm[iot])
            for iot, ot in enumerate(Ot)
        ]

        axs[i].set_title(str(np.round(100 / (2 * np.pi) * ph, decimals=1)) + " ms")

        axs[i].set_xlim((0, 0.6))
        axs[i].spines[["top"]].set_visible(False)
        axs[i].set_ylim((0, 1.0))

        ax2[i] = axs[i].twinx()
        ax2[i].plot(
            timevec[t:],
            alpha_params[1][0]
            * np.sin(2 * np.pi * alpha_params[0][0] * timevec[t:] + alpha_params[2][0])
            + alpha_params[1][0],
            color="k",
            linestyle="-.",
            linewidth=1.5,
        )
        ax2[i].plot(
            timevec[t:],
            alpha_params[1][1]
            * np.sin(2 * np.pi * alpha_params[0][1] * timevec[t:] + alpha_params[2][1])
            + alpha_params[1][1],
            color=np.array((0.5, 0.5, 0.5)),
            linestyle="-.",
            linewidth=1.5,
        )

        ax2[i].set_ylim((0, 1.0))

        if i == 0 or i == 3 or i == 6 or i == 9:
            axs[i].set_yticks((0, 0.5, 1))
            axs[i].set_ylabel("softm. activation")
        else:
            axs[i].set_yticks(())

        if i == 2 or i == 5 or i == 8 or i == 11:
            ax2[i].set_yticks((0, 0.5, 1))
            ax2[i].set_ylabel("a amplitude")
        else:
            ax2[i].set_yticks(())

        if i > 8:
            axs[i].set_xticks((0, 0.3, 0.6))
            axs[i].set_xlabel("time (s)")
        else:
            axs[i].set_xticks(())

        ax2[i].spines[["top"]].set_visible(False)

        axs[i].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    return fig
