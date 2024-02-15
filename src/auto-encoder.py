import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # Output: (16, 112, 112)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: (32, 56, 56)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: (64, 28, 28)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Output: (128, 14, 14)
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # Output: (256, 7, 7)
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # Output: (128, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # Output: (64, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # Output: (32, 56, 56)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Output: (16, 112, 112)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # Output: (3, 224, 224)
            nn.Sigmoid()  # Ensuring the output is within [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
