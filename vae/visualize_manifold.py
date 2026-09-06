"""
Visualise the VAE's 2D latent manifold as a grid of decoded images.
Only meaningful when latent_dim=2 (default in vae_model.py).

Also plots the encoded test set in latent space, coloured by nothing in
particular (OASIS has no class labels here) but useful to show the
distribution is roughly a standard normal, confirming the KL term worked.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from dataset import get_oasis_dataloaders
from vae_model import VAE


def plot_manifold_grid(model, device, out_path, n=15, digit_size=256, span=3.0):
    """
    Decode an n x n grid of points sampled across the latent space
    (using inverse-CDF spacing so the grid covers equal probability mass,
    same trick as the classic Keras VAE manifold example).
    """
    grid_x = norm.ppf(np.linspace(0.01, 0.99, n))
    grid_y = norm.ppf(np.linspace(0.01, 0.99, n))

    figure = np.zeros((digit_size * n, digit_size * n))

    model.eval()
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                decoded = model.decode(z).cpu().numpy().reshape(digit_size, digit_size)
                figure[i * digit_size:(i + 1) * digit_size,
                       j * digit_size:(j + 1) * digit_size] = decoded

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gray")
    plt.title("VAE Latent Manifold (2D grid sampling)")
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved manifold grid to {out_path}")


def plot_latent_scatter(model, loader, device, out_path, max_points=2000):
    """Encode test set slices and scatter their (mu_x, mu_y) positions."""
    model.eval()
    mus = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            mus.append(mu.cpu().numpy())
            if sum(len(m) for m in mus) >= max_points:
                break
    mus = np.concatenate(mus, axis=0)[:max_points]

    plt.figure(figsize=(8, 8))
    plt.scatter(mus[:, 0], mus[:, 1], s=4, alpha=0.5)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("Encoded test slices in latent space")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved latent scatter to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/groups/comp3710/OASIS")
    parser.add_argument("--checkpoint", type=str, default="./vae_out/vae_checkpoint.pt")
    parser.add_argument("--out_dir", type=str, default="./vae_out")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = VAE(latent_dim=ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if ckpt["latent_dim"] != 2:
        raise ValueError(
            f"Manifold grid visualisation needs latent_dim=2, checkpoint has "
            f"{ckpt['latent_dim']}. Use UMAP/t-SNE instead for higher dims."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    plot_manifold_grid(model, device, os.path.join(args.out_dir, "manifold_grid.png"))

    _, _, test_loader = get_oasis_dataloaders(args.data_root, batch_size=64)
    plot_latent_scatter(model, test_loader, device, os.path.join(args.out_dir, "latent_scatter.png"))