
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # Input: IR (1) + RGB (3) = 4 channels
            spectral_norm(nn.Conv2d(4, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, 4, 1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))
        )

    def forward(self, ir, rgb):

        # Concatenate IR and RGB along channel dimension
        x = torch.cat((ir, rgb), dim=1)

        return self.model(x)
