"""
Visualise UNet segmentation predictions and report final per-class DSC.
This is also what you'll run LIVE during your practical demo, per the brief's
"run live inference on a test set during demo" requirement.

Usage after training:
    python3 predict_unet.py --checkpoint unet_out/unet_best.pt
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_oasis_seg_dataloaders, NUM_CLASSES
from model import UNet
from train_unet import dice_per_class  # reuse the same Dice function


def visualize_predictions(model, loader, device, out_path, n=6):
    """
    Shows n examples: original slice | ground truth mask | predicted mask.
    Class colours are just matplotlib's default colormap indices -- consistent
    across the two mask columns so you can visually compare them.
    """
    model.eval()
    img, mask = next(iter(loader))
    img, mask = img[:n].to(device), mask[:n].to(device)

    with torch.no_grad():
        logits = model(img)
        pred = torch.argmax(logits, dim=1)       # (n, H, W) class indices
        target = torch.argmax(mask, dim=1)        # (n, H, W) class indices

    fig, axes = plt.subplots(3, n, figsize=(n * 2.2, 7))
    for i in range(n):
        axes[0, i].imshow(img[i, 0].cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(target[i].cpu(), cmap="viridis", vmin=0, vmax=NUM_CLASSES - 1)
        axes[1, i].axis("off")
        axes[2, i].imshow(pred[i].cpu(), cmap="viridis", vmin=0, vmax=NUM_CLASSES - 1)
        axes[2, i].axis("off")

    axes[0, 0].set_title("Input", loc="left", fontsize=10)
    axes[1, 0].set_title("Ground Truth", loc="left", fontsize=10)
    axes[2, 0].set_title("Prediction", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved prediction comparison to {out_path}")


def evaluate_test_set(model, loader, device):
    """Runs full test set, returns mean per-class Dice. This is the number
    you report as your final DSC result -- test set, not validation."""
    model.eval()
    dice_sum = torch.zeros(NUM_CLASSES)
    n_batches = 0
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            probs = F.softmax(logits, dim=1)
            dice_sum += dice_per_class(probs, mask).cpu()
            n_batches += 1
    return (dice_sum / n_batches).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/groups/comp3710/OASIS")
    parser.add_argument("--checkpoint", type=str, default="./unet_out/unet_best.pt")
    parser.add_argument("--out_dir", type=str, default="./unet_out")
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    _, _, test_loader = get_oasis_seg_dataloaders(
        args.data_root, batch_size=16, image_size=args.image_size
    )

    os.makedirs(args.out_dir, exist_ok=True)
    visualize_predictions(model, test_loader, device,
                           os.path.join(args.out_dir, "predictions.png"))

    print("\nEvaluating full test set...")
    test_dice = evaluate_test_set(model, test_loader, device)
    for i, d in enumerate(test_dice):
        status = "PASS" if d > 0.9 else "FAIL"
        print(f"  Class {i}: DSC = {d:.4f}  [{status}]")
    print(f"  Mean DSC: {test_dice.mean():.4f}")
    print(f"  All classes > 0.9: {all(d > 0.9 for d in test_dice)}")