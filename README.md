# Dynamical Artificial Neural Network embracing the temporal dynamics of the ventral stream

for pre-print, see: https://www.biorxiv.org/content/10.1101/2023.11.27.568876v1

use DynANN.ipynb to run code and reproduce figures in manuscript

### /aet_pytorch
contains all scripts to implement and train a neural network on the AET problem

aet_net: code to implement network structure for 1-layer network (a 2-layer network did a better job at segregating competing inputs, so this is not explored in manuscript)

aet_net_2lay: code to implement network structure for 2-layer network

aet_stim: generate visual inputs and minibatches

aet_dyn: Euler integration as functions (however, these are explicitly coded in current notebook)

aet_optuna_bear: exploratory script, finding tunable parameters with optuna (optimized for HPC)

*work in progress*
mnist_net: extension of ideas to network trained on MNIST
mnist_stim: create mini batches of MNIST stimuli



