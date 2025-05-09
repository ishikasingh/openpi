#!/bin/bash

# Initialize conda
source $HOME/miniconda3/etc/profile.d/conda.sh || source $HOME/anaconda3/etc/profile.d/conda.sh

# Activate the specified environment
conda activate $1

# Execute the remaining arguments
"${@:2}" 