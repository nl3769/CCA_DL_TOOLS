""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class GAN_UPSAMPLE_02(nn.Module):
    def __init__(self, p, n_channels=1, n_classes=1, bilinear=False):
        super(GAN_UPSAMPLE_02, self).__init__()

        self.inc = DoubleConv(n_channels, 64, p)
        self.down1 = Down(64, 128, p)
        self.down2 = Down(128, 256, p)
        self.down3 = Down(256, 512, p)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, p)
        self.up1 = Up(1024, 512 // factor, p)
        self.up2 = Up(512, 256 // factor, p)
        self.up3 = Up(256, 128 // factor, p)
        self.up4 = Up(128, 64, p)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits

# ----------------------------------------------------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, p, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=p.KERNEL_SIZE, padding='same', bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=p.KERNEL_SIZE, padding='same', bias=True),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# ----------------------------------------------------------------------------------------------------------------------
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# ----------------------------------------------------------------------------------------------------------------------
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, p):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if p.BILINEAR:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=p.KERNEL_SIZE, padding='same'))
            self.conv = DoubleConv(in_channels, out_channels, p, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ----------------------------------------------------------------------------------------------------------------------
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
