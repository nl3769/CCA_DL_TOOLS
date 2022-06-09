'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import torch
import torch.nn                             as  nn


# ----------------------------------------------------------------------------------------------------------------------
class DilatedUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedUnet, self).__init__()

        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)

        self.bottleneck = bottleneck(512, 128)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = FinalLayer(128, out_channels)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, I1, I2):

        x = torch.cat((I1, I2), 1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # b0 = self.bottleneck(d5)

        u1 = self.up1(d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        out = self.final(u4, d1)
        M1, M2 = torch.chunk(out, 2, 1)

        return M1, M2

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
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2)
          )

    # ------------------------------------------------------------------------------------------------------------------
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
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4,
                               stride=2, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x

class bottleneck(nn.Module):

    def __init__(self, in_size, out_size):
        super(bottleneck, self).__init__()

        self.convd1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding='same', dilation=1)
        self.convd2 = nn.Conv2d(in_size, out_size, kernel_size=3, padding='same', dilation=2)
        self.convd3 = nn.Conv2d(in_size, out_size, kernel_size=3, padding='same', dilation=3)
        self.convd4 = nn.Conv2d(in_size, out_size, kernel_size=3, padding='same', dilation=4)

        self.batchNorm1 = nn.BatchNorm2d(out_size, affine=True)
        self.batchNorm2 = nn.BatchNorm2d(out_size, affine=True)
        self.batchNorm3 = nn.BatchNorm2d(out_size, affine=True)
        self.batchNorm4 = nn.BatchNorm2d(out_size, affine=True)

        self.relu = nn.ReLU()

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):

        out1 = self.convd1(x)
        out1 = self.batchNorm1(out1)

        out2 = self.convd2(x)
        out2 = self.batchNorm2(out2)

        out3 = self.convd3(x)
        out3 = self.batchNorm3(out3)

        out4 = self.convd4(x)
        out4 = self.batchNorm4(out4)

        out = torch.cat((out1, out2, out3, out4), 1)
        out = self.relu(out)

        return out

    # ------------------------------------------------------------------------------------------------------------------

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
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x

# ----------------------------------------------------------------------------------------------------------------------