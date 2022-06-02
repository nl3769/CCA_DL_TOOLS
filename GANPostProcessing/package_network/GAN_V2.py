import torch
import torch.nn as nn

def discriminator_block(in_filters, out_filters):

    """ Return down sampling layers of each discriminator block. """

    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=(4, 4), padding=1)]
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

# ----------------------------------------------------------------------------------------------------------------------
class Discriminator(nn.Module):

    def __init__(self, in_channels=1):

        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            discriminator_block(in_channels*2, 64),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
            nn.Conv2d(512, 1, 1, stride=(2, 1), padding=0),
        )


    def forward(self, img_A, img_B):
        # --- Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.layers(img_input)

# ----------------------------------------------------------------------------------------------------------------------
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.firstLayer = firstLayer(in_channels, 32)

        self.down1 = UNetDown(32, 64)
        self.down2 = UNetDown(64, 96)
        self.down3 = UNetDown(96, 192)
        self.down4 = UNetDown(192, 384)

        self.bottlneck = UNetDown(384, 384)

        self.up1 = UNetUp(768, 192)
        self.up2 = UNetUp(384, 96)
        self.up3 = UNetUp(192, 64)
        self.up4 = UNetUp(128, 32)

        self.final = FinalLayer(32, out_channels)

    def forward(self, x):

        x = self.firstLayer(x)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottlneck = self.bottlneck(d4)

        # upbottlneck = self.upbottlneck(bottlneck)

        u1 = self.up1(bottlneck, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        out = self.final(u4, x)

        return out

# ----------------------------------------------------------------------------------------------------------------------
class UNetDown(nn.Module):
    """Descending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.AvgPool2d(2, stride=2, padding=0),

            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
          )

    def forward(self, x):
        return self.model(x)

# ----------------------------------------------------------------------------------------------------------------------
class UNetUp(nn.Module):
    """Ascending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """

    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input=None):

        x = nn.Upsample(scale_factor=2)(x)
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x

# ----------------------------------------------------------------------------------------------------------------------
class FinalLayer(nn.Module):
    """Final block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size):

        super(FinalLayer, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_size * 2, in_size, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(in_size),
            nn.ReLU(),
            nn.Conv2d(in_size, in_size, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(in_size),
            nn.ReLU(),
            nn.Conv2d(in_size, out_size, kernel_size=7, stride=1, padding=3),
        )

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x, skip_input):
        x = nn.Upsample(scale_factor=2)(x)
        x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)

        return x

# ----------------------------------------------------------------------------------------------------------------------
class firstLayer(nn.Module):
    """First block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size):

        super(firstLayer, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        x = self.model(x)
        return x