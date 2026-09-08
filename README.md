# COMP3710 — Lab Demo 2, Part 4: Recognition Problems

**Student:** s4911688
**Tier attempted:** Medium (Task 1: VAE, Task 2: UNet)

## What's in here

This repo covers Part 4 of Lab Demo 2 — two pattern recognition models trained
on the preprocessed OASIS brain MRI dataset, which lives at
`/home/groups/comp3710/OASIS/` on Rangpur. Task 3 (GAN) was deliberately left
out to protect time for getting Tasks 1 and 2 done properly rather than
rushing all three.

```
comp3710-A2-part4-recognition/
├── vae/
│   ├── dataset.py           # loads raw OASIS slices
│   ├── vae_model.py         # encoder/decoder + VAE loss
│   ├── train_vae.py         # training loop
│   ├── visualize_manifold.py # latent space grid + scatter plot
│   ├── submit_vae.slurm     # Rangpur job script
│   └── vae_out/             # loss curve, manifold plots, checkpoint (gitignored)
├── unet/
│   ├── dataset.py           # loads paired image + segmentation mask
│   ├── model.py              # UNet architecture
│   ├── train_unet.py        # training loop with Dice loss
│   ├── predict_unet.py      # inference + visualization for the live demo
│   ├── submit_unet.slurm    # Rangpur job script
│   └── unet_out/             # loss/Dice curves, predictions, checkpoints (gitignored)
└── README.md
```

## Task 1 — Variational Autoencoder

The idea: compress each brain slice down to just 2 numbers (the latent
space), then rebuild the slice from those 2 numbers. If it rebuilds well, it
means the model has learned what actually makes a brain scan look like a
brain scan, not just memorised individual images.

Using a 2D latent space specifically (rather than something bigger) means we
can plot it directly as a grid — decode a sweep of points across that 2D
space and see how the images morph as you move through it. That's the
"manifold" the brief asks for.

**Results:** trained for 20 epochs, loss converged cleanly (4717 → 4182
train loss). The manifold grid shows a smooth, believable sweep across brain
shapes. The latent scatter plot shows encoded slices clustering slightly off
from the origin rather than sitting in a perfect standard normal — a known
and minor side effect of prioritising reconstruction quality, doesn't affect
the manifold visualisation.

```bash
cd vae
sbatch submit_vae.slurm
```

## Task 2 — UNet Segmentation

This one takes a brain slice and labels every single pixel with which
anatomical class it belongs to — 4 classes total (background + 3 tissue
types), based on the mask values found in the OASIS segmentation data
(`0, 85, 170, 255`, remapped to class indices `0–3`).

The architecture is a standard UNet: an encoder that shrinks the image down
while learning what's present, a decoder that grows it back up while
learning where things are, and skip connections between matching resolution
levels so the decoder doesn't lose the fine boundary detail the encoder saw
early on. Trained with Dice loss instead of plain cross-entropy, since Dice
directly optimises the overlap metric (DSC) the task is actually graded on —
cross-entropy can look fine on paper while still doing badly on the smaller
classes.

**Target:** DSC > 0.9 for every one of the 4 classes on the test set.

```bash
cd unet
sbatch submit_unet.slurm
# once training finishes:
python3 predict_unet.py --checkpoint unet_out/unet_best.pt
```

`predict_unet.py` saves a side-by-side comparison (input / ground truth /
prediction) and prints a pass/fail per class against the 0.9 threshold —
this is also what gets run live during the practical demo.

## A couple of things worth knowing if you're marking or reading this

- Mask resizing uses **nearest-neighbour** interpolation, not the default.
  Regular interpolation blends adjacent class values together and creates
  pixel values that don't correspond to any real class — an easy mistake to
  make and hard to notice unless you check for it.
- The dataset loader checks that every image has a matching mask *before*
  training starts, so a broken pairing fails loudly and immediately instead
  of quietly training on mismatched data.

## AI Use Disclosure

See `prompt_history.md` for a log of AI assistance used during development.