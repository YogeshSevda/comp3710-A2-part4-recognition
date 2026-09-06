#!/bin/bash
#SBATCH --job-name=vae_oasis
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=vae_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

python train_vae.py --epochs 20 --batch_size 64 --latent_dim 2 --image_size 128
python visualize_manifold.py --latent_dim 2 --image_size 128 --checkpoint ./vae_out/vae_checkpoint.pt