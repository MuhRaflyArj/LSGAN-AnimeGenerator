"""Generator and discriminator networks for the anime GAN models."""

from torch import nn
from torch.nn.utils import spectral_norm


def weight_init_normal(m):
    """Apply normal initialization to supported layers."""

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """Map latent vectors to 64x64 RGB anime face images."""

    def __init__(self, latent_dim=128, channels=3):
        """Build the generator network."""

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.Unflatten(1, (512, 4, 4)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """Generate an image batch from latent vectors."""

        return self.network(x)


class Discriminator(nn.Module):
    """Score input images as real or fake."""

    def __init__(self, channels=3):
        """Build the discriminator network."""

        super().__init__()
        self.network = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adv_layer = spectral_norm(nn.Linear(512 * 4 * 4, 1))

    def forward(self, x):
        """Return discriminator logits for a batch of images."""

        x = self.network(x)
        x = x.view(x.shape[0], -1)
        return self.adv_layer(x)
