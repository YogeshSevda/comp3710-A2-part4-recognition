#!/bin/bash
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --job-name=vae_oasis
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=vae_%j.out
#SBATCH --error=vae_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

cd ~/comp3710_part4/vae   # adjust if you upload elsewhere

python train_vae.py \
    --data_root /home/groups/comp3710/OASIS \
    --out_dir ./vae_out \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --latent_dim 2 \
    --beta 1.0

python visualize_manifold.py \
    --data_root /home/groups/comp3710/OASIS \
    --checkpoint ./vae_out/vae_checkpoint.pt \
    --out_dir ./vae_out
