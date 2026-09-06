#!/bin/bash
#SBATCH --job-name=unet_oasis
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=unet_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

python train_unet.py --epochs 30 --batch_size 16 --image_size 128
