import optuna
import torch
from torch import nn

import numpy as np
import aet_net
from itertools import combinations
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model parameters

nn_dim_ = [28, 68, 3]  # [quadrant size, number of hidden nodes, number of output nodes]
mini_sz_ = 1  # mini batch size (1 = use SGD)
num_epo_ = 80

lossfun = [
    nn.MSELoss(),
    nn.Softmax(dim=-1),
]  # loss function & final layer activation (for binary crossentropy use sigmoid)

data, output = aet_net.aet_stim.mkstim()  # load data for the tangledness test


def objective(trial):

    # to be optimized params
    eta_ = trial.suggest_float("ETA_", 0.001, 0.2)
    beta_ = trial.suggest_float("BETA_", 0.001, 0.01)
    p_ = trial.suggest_float("P_", 1e-4, 2e-2)
    slope_sigm_ = trial.suggest_int("slope_sig", 2, 5, step=0.5)

    kl_reg_ = [beta_, p_]  # identified with optuna

    sig_param = [slope_sigm_, 0]  # sigmoid slope and shift in x direction

    params = nn_dim_, eta_, mini_sz_, num_epo_, kl_reg_, sig_param

    # initialize model and weights
    model = aet_net.net(params, lossfun)
    model = aet_net.init_params(model, weight_init="uni")
    optimizer = torch.optim.SGD(model.parameters(), lr=eta_)

    model.to(DEVICE)
    loss_hist = model.train(optimizer, noise=False, print_loss=False)

    # "tangledness" of hidden represenations

    idx = np.array((0, 5, 10))  # ,-1))
    inp_combi = list(combinations(idx, 2))  # possible input combinations
    all_angle_sum = torch.zeros(len(inp_combi))

    for i, c in enumerate(inp_combi):

        H1 = data[c[0]]
        H2 = data[c[1]]
        H3 = data[c[0]] + data[c[1]]

        # apply layer
        _, H1, _ = model.forw_conv(H1)
        _, H2, _ = model.forw_conv(H2)
        _, H3, _ = model.forw_conv(H3)

        H1_2 = H1 + H2

        num_ = torch.matmul(H3, H1_2)
        denom_ = torch.linalg.vector_norm(H3) * torch.linalg.vector_norm(H1_2)
        all_angle_sum[i] = (torch.acos(num_ / denom_) * 180 / torch.pi).cpu().detach()

    return torch.mean(loss_hist[:-20]) + torch.mean(all_angle_sum)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300, timeout=600)

# pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
# complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

# print("Study statistics: ")
# print("  Number of finished trials: ", len(study.trials))
# print("  Number of pruned trials: ", len(pruned_trials))
# print("  Number of complete trials: ", len(complete_trials))

# print("Best trial:")
# trial = study.best_trial

# print("  Value: ", trial.value)

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))


optuna_aet_result = {key: value for key, value in study.best_trial.params.items()}
optuna_aet_result = [{"loss": study.best_trial.value}, optuna_aet_result]

with open("optuna_aet_trial.pkl", "wb") as fp:
    pickle.dump(optuna_aet_result, fp)
    print("dictionary saved successfully to file")
