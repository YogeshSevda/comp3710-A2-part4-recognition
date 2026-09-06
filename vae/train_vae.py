"""
Train the VAE on OASIS brain MRI slices.
Run on Rangpur via sbatch (see submit_vae.slurm) so it lands on a GPU node.
"""

import os
import argparse
import time
import torch
import matplotlib
matplotlib.use("Agg")  # no display on compute nodes
import matplotlib.pyplot as plt

from dataset import get_oasis_dataloaders
from vae_model import VAE, vae_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_oasis_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=4
    )

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_samples += batch.size(0)

        avg_train_loss = running_loss / n_samples
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_running = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss, _, _ = vae_loss(recon, batch, mu, logvar, beta=args.beta)
                val_running += loss.item()
                val_n += batch.size(0)
        avg_val_loss = val_running / val_n
        val_losses.append(avg_val_loss)

        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | train_loss/sample: {avg_train_loss:.2f} "
              f"| val_loss/sample: {avg_val_loss:.2f} | time: {dt:.1f}s")

        # Save checkpoint each epoch (cheap insurance against job timeouts)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "latent_dim": args.latent_dim,
        }, os.path.join(args.out_dir, "vae_checkpoint.pt"))

    # Save loss curve
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss per sample")
    plt.title("VAE training loss")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150)
    print(f"Saved loss curve to {args.out_dir}/loss_curve.png")
    print(f"Saved final checkpoint to {args.out_dir}/vae_checkpoint.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/groups/comp3710/OASIS")
    parser.add_argument("--out_dir", type=str, default="./vae_out")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    train(args)