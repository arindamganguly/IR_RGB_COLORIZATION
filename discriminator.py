import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        # Input = IR (1 channel) + RGB (3 channel) = 4 channels
        self.model = nn.Sequential(

            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, ir, rgb):

        # combine IR + RGB
        x = torch.cat((ir, rgb), dim=1)

        return self.model(x)