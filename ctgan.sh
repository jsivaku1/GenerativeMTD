#!/bin/bash -l
# Job Name
#SBATCH --job-name=CTGAN
# Output File Name
#SBATCH --output=CTGAN_output.txt
# Error File Name
#SBATCH --error=CTGAN_error.log
# Number of Nodes to Use
#SBATCH --nodes=1
# Number of Tasks per Node
#SBATCH --tasks-per-node=1
# Which Partition to Use
#SBATCH --partition=gpucompute
# Memory to allocate in Each Node
#SBATCH --mem=30GB
# Number of GPUs to use per Node
#SBATCH --gres=gpu:2

# Load module
module load cuda11.1/toolkit/11.1.1

# Activate Conda Env
conda activate deepmtd




python3 train.py --model 'ctgan' --dataset 'Data/imputed_SweatBinary.csv' --target_col_ix 18 --k 3 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/cleveland_heart.csv' --target_col_ix 13 --k 5 --num_obs 10 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/urban_land.csv' --target_col_ix 0 --k 10 --num_obs 10 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/mammography.csv' --target_col_ix 5 --k 5 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/immunotherapy.csv' --target_col_ix 7 --k 8 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/cryotherapy.csv' --target_col_ix 6 --k 6 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/caesarian.csv' --target_col_ix 5 --k 6 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/cervical.csv' --target_col_ix 19 --k 3 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/breast.csv' --target_col_ix 9 --k 3 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/post_operative.csv' --target_col_ix 8 --k 6 --num_obs 100 --epochs 200



python3 train.py --model 'ctgan' --dataset 'Data/community_crime.csv' --target_col_ix 122 --ml_utility regression --k 5 --num_obs 10 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/sweat_ordinal.csv' --target_col_ix 18 --ml_utility regression --k 3 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/fertility.csv' --target_col_ix 8 --ml_utility regression --k 4 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/parkinsons.csv' --target_col_ix 22 --ml_utility regression --k 3 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/thyroid.csv' --target_col_ix 5 --ml_utility regression --k 6 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/liver.csv' --target_col_ix 1 --ml_utility regression --k 4 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/fat.csv' --target_col_ix 0 --ml_utility regression --k 7 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/pima.csv' --target_col_ix 6 --ml_utility regression --k 3 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/prostate.csv' --target_col_ix 8 --ml_utility regression --k 4 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/bioconcentration.csv' --target_col_ix 10 --ml_utility regression --k 3 --num_obs 100 --epochs 200

python3 train.py --model 'ctgan' --dataset 'Data/heartfail.csv' --target_col_ix 12 --ml_utility regression --k 3 --num_obs 100 --epochs 200