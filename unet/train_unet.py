"""
Train the UNet on OASIS brain MRI segmentation (Task 2).
Run on Rangpur via sbatch (see submit_unet.slurm).

Loss: Dice loss, directly optimising the metric we're graded on (DSC),
rather than plain cross-entropy which can look "good" while still doing
badly on small minority classes.
"""

import os
import argparse
import time
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_oasis_seg_dataloaders, NUM_CLASSES
from model import UNet


def dice_per_class(pred_probs, target_onehot, eps=1e-6):
    """
    pred_probs: (B, C, H, W) softmax probabilities
    target_onehot: (B, C, H, W) one-hot ground truth
    Returns: tensor of shape (C,) -- Dice score per class, averaged over batch.

    Dice = 2 * |intersection| / (|pred| + |target|)
    A score of 1.0 = perfect overlap, 0.0 = no overlap at all.
    """
    dims = (0, 2, 3)  # sum over batch, height, width -- keep classes separate
    intersection = torch.sum(pred_probs * target_onehot, dim=dims)
    union = torch.sum(pred_probs, dim=dims) + torch.sum(target_onehot, dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice  # shape (C,)


def dice_loss(logits, target_onehot):
    """1 - mean Dice across classes. Lower is better, converges toward 0."""
    probs = F.softmax(logits, dim=1)
    dice_scores = dice_per_class(probs, target_onehot)
    return 1.0 - dice_scores.mean()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_oasis_seg_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=1, image_size=args.image_size
    )

    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    train_losses, val_losses = [], []
    val_dice_history = []  # list of per-class dice arrays, one per epoch

    best_mean_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = dice_loss(logits, mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation: track loss AND per-class Dice
        model.eval()
        val_running = 0.0
        dice_sum = torch.zeros(NUM_CLASSES)
        n_batches = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                loss = dice_loss(logits, mask)
                val_running += loss.item() * img.size(0)

                probs = F.softmax(logits, dim=1)
                dice_sum += dice_per_class(probs, mask).cpu()
                n_batches += 1

        avg_val_loss = val_running / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        avg_dice = (dice_sum / n_batches).numpy()
        val_dice_history.append(avg_dice)

        dt = time.time() - t0
        dice_str = " ".join(f"C{i}={d:.3f}" for i, d in enumerate(avg_dice))
        print(f"Epoch {epoch}/{args.epochs} | train_loss: {avg_train_loss:.4f} "
              f"| val_loss: {avg_val_loss:.4f} | val_dice: [{dice_str}] "
              f"| mean_dice: {avg_dice.mean():.4f} | time: {dt:.1f}s")

        # Save best model by mean validation Dice
        if avg_dice.mean() > best_mean_dice:
            best_mean_dice = avg_dice.mean()
            torch.save(model.state_dict(), os.path.join(args.out_dir, "unet_best.pt"))

    # Always also save the final epoch's weights
    torch.save(model.state_dict(), os.path.join(args.out_dir, "unet_final.pt"))

    # Plot loss curve
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Dice loss")
    plt.title("UNet training loss")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150)

    # Plot per-class Dice over training
    val_dice_history = torch.tensor(val_dice_history)  # (epochs, C)
    plt.figure()
    for c in range(NUM_CLASSES):
        plt.plot(val_dice_history[:, c], label=f"Class {c}")
    plt.axhline(0.9, color="red", linestyle="--", label="DSC=0.9 target")
    plt.xlabel("Epoch")
    plt.ylabel("Validation DSC")
    plt.title("Per-class Dice Score over training")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "dice_curve.png"), dpi=150)

    print(f"\nBest mean validation Dice: {best_mean_dice:.4f}")
    print(f"Final per-class Dice: {val_dice_history[-1].tolist()}")
    print(f"Saved: {args.out_dir}/unet_best.pt, unet_final.pt, loss_curve.png, dice_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/groups/comp3710/OASIS")
    parser.add_argument("--out_dir", type=str, default="./unet_out")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()

    train(args)