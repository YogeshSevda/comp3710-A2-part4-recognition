# AI Prompt / Assistance History

## Part 4, Task 1 — VAE
- Used Claude (Anthropic) to scaffold the initial VAE architecture
  (encoder/decoder conv structure), training loop, and latent manifold
  visualization code (2D grid sweep + latent scatter plot).
- Discussed the trade-off between a 2D latent space (directly plottable,
  slightly blurrier reconstructions) and a higher-dimensional one (better
  reconstructions, needs UMAP/dim-reduction to visualize) — chose 2D for a
  clearer, more explainable manifold picture for the live demo.
- Iterated on Rangpur job setup (Slurm script, conda environment, dataset
  path) with AI assistance for debugging environment/config issues
  (missing matplotlib install, stale file paths in the job script).
- Final training run, hyperparameters, and result verification performed
  by the student on Rangpur; all outputs (loss curve, manifold grid, latent
  scatter) generated from real training runs, not fabricated.

## Part 4, Task 2 — UNet
- Used Claude to design the paired image/mask dataset loader, including
  the class-value remapping (raw mask values 0/85/170/255 -> class indices
  0-3) and the one-hot encoding required by the brief.
- Discussed and applied nearest-neighbour interpolation for mask resizing
  specifically, to avoid blending adjacent class values into invalid
  "classes" during resizing (a subtlety flagged during development).
- Used Claude to scaffold the UNet architecture (encoder/decoder with skip
  connections) and the training loop, including per-class Dice loss and
  per-epoch DSC tracking against the >0.9 threshold required by the brief.
- Used Claude to write the inference/visualization script used for the
  live demo (predictions image + per-class pass/fail report).
- Final training run on Rangpur, DSC results (0.9995 / 0.9447 / 0.9524 /
  0.9729 per class on the test set), and verification performed by the
  student; all results are from real training runs on the OASIS dataset.

## General notes
- AI assistance was used for code scaffolding, debugging Rangpur/Slurm/git
  workflow issues, and explaining concepts (VAE latent space, Dice loss,
  skip connections) for the student's own understanding ahead of the oral
  demo. All architectural decisions, hyperparameter choices, and final
  results were reviewed, run, and verified by the student on Rangpur.