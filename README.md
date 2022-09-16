# dnn_osci

Integrating oscillations into neural networks for multiplexinb.
Note that these scripts are work in progress

/manual: network with one hidden layer, implemented without API
  AET_model.py: module: network implemented using automatic differentiation & matrix multiplication 
  AETZ_train.py: script training simple model on 4 stimuli, creates competitive network, plots to explore dynamics
 
/aet_pytorch: manual network replicated (fully) in Pytorch (for GPU support)

/CSHL project: integrating dynamics into CORnet-Z (Kubilius et al., 2019, NeurIPS)

/mnist: network with oscillations trained on MNIST data set 

/old: archived scripts
