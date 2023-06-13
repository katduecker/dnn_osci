#!/bin/bash
#SBATCH --time 1:00:0
#SBATCH --qos bbdefault
#SBATCH --account=jenseno-visual-search-rft


set -eu

module purge; module load bluebear

module load PyTorch/1.10.0-foss-2021a
module load Optuna/2.9.1-foss-2021a
module load torchvision/0.11.1-foss-2021a-PyTorch-1.10.0


python3 aet_optuna_bear.py