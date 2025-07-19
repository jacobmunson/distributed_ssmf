#!/bin/bash
source /etc/profile

# Purge all loaded modules to avoid conflicts
module purge

# Load required modules
module load GCCcore/13.2.0
module load GCC/13.2.0
module load OpenMPI/4.1.6-GCC-13.2.0

# Source your bash config (for conda)
source ~/.bashrc

# Activate the desired conda environment
conda activate DistributedSSMF_env
