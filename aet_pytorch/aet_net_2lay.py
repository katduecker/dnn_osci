# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# -*- coding: utf-8 -*-

import torch
from torch import nn
import aet_stim


# sigmoid activation with option to stretch and shift
def sigmoid(z, sig_param):

    _slope, _bias, _ = sig_param

    return 1.0 / (1.0 + torch.exp(-_slope * (z + _bias)))


def CE_loss(output_hat, output_):

    return -torch.sum(output_ * torch.log(output_hat))


# weight & bias initialization
def init_params(model, weight_init="normal"):

    for m in model.modules():
        # if module is a conv or linear layer, set weights
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if weight_init == "uni":
                nn.init.xavier_uniform_(m.weight)  # ,mean=0,std=0.2)
            elif weight_init == "normal":
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif weight_init == "zero":
                nn.init.constant_(m.weight, 0.1)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return model


class net(nn.Module):

    def __init__(self, params, lfun):

        super(net, self).__init__()

        dims, lr, mini_sz, num_ep, reg, sig_param, lmbda, lay_regu = params

        ## NETWORK ARCHITECTURE

        # convolutional & fully connected layer
        if sig_param[0][2]:  # when using set bias, don't learn
            self.conv1 = nn.Conv2d(1, dims[1], dims[0], stride=dims[0], bias=True)
        else:
            self.conv1 = nn.Conv2d(1, dims[1], dims[0], stride=dims[0], bias=True)

        if sig_param[1][2]:  # when using set bias, don't learn

            self.fc1 = nn.Linear(dims[1], dims[2], bias=True)
            self.fc2 = nn.Linear(dims[2], dims[-1], bias=True)
        else:
            self.fc1 = nn.Linear(dims[1], dims[2], bias=False)
            self.fc2 = nn.Linear(dims[2], dims[-1], bias=False)

        self.acti1 = sigmoid
        self.pool1 = torch.sum
        self.sig_param = sig_param
        self.lmbda = lmbda
        self.lay_regu = lay_regu
        self.dims = dims

        # flatten
        self.flat = nn.Flatten()

        # Fully connected layer
        self.actiout = lfun[1]

        ## TRAINING PARAMETERS
        # loss function
        self.lossfun = lfun[0]

        # regularizer
        self.reg = reg

        # mini batch size & number of epochs
        self.mini_sz = mini_sz
        self.num_ep = num_ep

    # forward sweep
    def forw_conv(self, data):

        ## convolutional layer
        y = self.conv1(data)

        # pool over quadrants
        Z = self.pool1(y, dim=(-2, -1))

        # activation
        H1 = sigmoid(Z, self.sig_param[0])

        H2 = sigmoid(self.fc1(H1), self.sig_param[1])

        # fully connect layer and activation
        O = self.actiout(self.fc2(H2))

        return Z, H1, H2, O

    # sparsity regularizer https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    def bias_regularizer(self, data):

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _beta, _rho, _lay = self.reg

        H = self.forw_conv(data)

        # calculate hidden activations
        rho_hat = H[_lay].sum(dim=0).to(DEVICE)

        rho_hat /= data.shape[0]

        # regularizer: KL divergence between hidden activations & rho (added to loss)
        regu_loss = _beta * torch.sum(
            _rho * torch.log(_rho / rho_hat)
            + (1 - _rho) * torch.log((1 - _rho) / (1 - rho_hat))
        )

        # penalty bias gradient
        regu_bias = _beta * (-_rho / rho_hat + ((1 - _rho) / (1 - rho_hat)))

        return regu_loss, regu_bias

    def ortho_regularizer(self, data):

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = self.forw_conv(data)

        H = x[self.lay_regu]

        p = torch.mean(
            self.flat((torch.mm(H.T, H) - torch.eye(H.shape[1])).unsqueeze(0) ** 2)
        )

        return self.lmbda * p

    def train(self, optimizer, dataset="aet", noise=False, print_loss=True):

        lay_sparse_penal = ["conv1.bias", "fc1.bias", "fc2.bias"]
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dataset == "aet":
            data, output = aet_stim.mkstim(noise)
        else:
            data, output = dataset

        data = data.to(DEVICE)
        output = output.to(DEVICE)

        loss = torch.zeros(
            (self.num_ep),
        ).to(DEVICE)

        for e in range(self.num_ep):

            mini_idx = aet_stim.make_minib(data.shape[0], mini_sz=self.mini_sz)

            for mb in range(len(mini_idx)):
                optimizer.zero_grad()
                # forward
                _, _, _, y = self.forw_conv(data[mini_idx[mb]])

                # regularizer (if sparsity params are defined)
                if self.reg:
                    _regu = self.bias_regularizer(data)
                else:
                    _regu = torch.zeros(2)

                lay_regu = self.ortho_regularizer(data)

                # loss + sparsity penalty
                _loss = self.lossfun(output[mini_idx[mb]], y) + _regu[0] + lay_regu

                # accumulate gradients for minibatch
                _loss.backward()

                # add sparsity penalty to bias
                if self.reg[0]:
                    bias = self.get_parameter(lay_sparse_penal[self.reg[2] - 1])
                    bias.grad += _regu[1]

                # update after mini batch
                optimizer.step()

            loss[e] = _loss  # .mean()

            if print_loss:
                print(f"epoch: {e}, loss: {loss[e]}")

        del data, output

        return loss
