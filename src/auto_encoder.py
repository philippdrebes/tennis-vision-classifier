import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.dropout = nn.Dropout(0.5)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.dropout,

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.dropout,

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.dropout,

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.dropout,
        )

        self.flattened_size = 256 * (224 // (2 ** 4)) ** 2  # Adjust based on the encoder output

        self.fc = nn.Linear(self.flattened_size, 4096)
        self.fc_relu = nn.ReLU()

        # Decoder
        self.decoder_fc = nn.Linear(4096, self.flattened_size)
        self.decoder_fc_relu = nn.ReLU()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_relu(self.fc(x))

        x = self.decoder_fc_relu(self.decoder_fc(x))
        x = x.view(-1, 256, (224 // (2 ** 4)), (224 // (2 ** 4)))  # Adjust based on the encoder output

        x = self.decoder(x)
        return x


# net = ConvAutoencoder()
# dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1
# output = net(dummy_input)
# print(output.size())  # Should be torch.Size([1, 3, 224, 224])
