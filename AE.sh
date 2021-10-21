#!/bin/bash -l
# Job Name
#SBATCH --job-name=AE
# Output File Name
#SBATCH --output=AE_output.txt
# Error File Name
#SBATCH --error=AE_error.log
# Number of Nodes to Use
#SBATCH --nodes=1
# Number of Tasks per Node
#SBATCH --tasks-per-node=2
# Which Partition to Use
#SBATCH --partition=gpucompute
# Memory to allocate in Each Node
#SBATCH --mem=30GB
# Number of GPUs to use per Node
#SBATCH --gres=gpu:2

# Load module
#module load cuda11.1/toolkit/11.1.1

# Activate Conda Env
#conda activate deepmtd

python3 train.py --model 'AE' --file 'Data/imputed_SweatBinary.csv' --k 3 --num_obs 100 --epochs 150
