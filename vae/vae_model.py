"""
Convolutional VAE for 256x256 grayscale OASIS brain MRI slices.

latent_dim defaults to 2 so we can directly visualise the manifold as a
2D grid (per the brief: "2D sampling grid or UMAP/dim-reduction").
A 2-D bottleneck is aggressive for 256x256 images, so reconstructions
will be blurry — that's expected and fine for this task; the point is
the manifold structure, not photorealistic reconstruction.
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=2, img_channels=1):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),  # 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),            # 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),           # 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),          # 16
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),          # 8
            nn.ReLU(),
        )
        self.flatten_dim = 256 * 8 * 8

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder: mirror of encoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),  # 16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),  # 256
            nn.Sigmoid(),  # output in [0, 1] to match ToTensor() input range
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Reconstruction loss (BCE, since output is sigmoid in [0,1]) + KL divergence.
    beta: weight on KL term (beta-VAE style). beta=1.0 is the standard VAE.
    Returns total loss, recon loss, KL loss (all summed over batch, not averaged,
    so you can divide by batch size yourself for logging).
    """
    recon_loss = nn.functional.binary_cross_entropy(
        recon_x, x, reduction="sum"
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss