import torch
import torch.nn as nn

class GeneratorUNetUPCONV(nn.Module):
    def __init__(self, p, in_channels=1, out_channels=1):
        super(GeneratorUNetUPCONV, self).__init__()

        self.down1 = UNetDown(in_channels, 64, p)
        self.down2 = UNetDown(64, 128, p)
        self.down3 = UNetDown(128, 256, p)
        self.down4 = UNetDown(256, 512, p)
        self.down5 = UNetDown(512, 512, p)

        self.up1 = UNetUp(512, 512, p)
        self.up2 = UNetUp(1024, 256, p)
        self.up3 = UNetUp(512, 128, p)
        self.up4 = UNetUp(256, 64, p)

        self.final = FinalLayer(128, out_channels, p)

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        out = self.final(u4, d1)

        return out

# ----------------------------------------------------------------------------------------------------------------------
class UNetDown(nn.Module):
    """Descending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size, p):
        super(UNetDown, self).__init__()

        self.k_size = p.KERNEL_SIZE
        self.normalization = p.NORMALIZATION

        self.conv2D = nn.Conv2d(in_size, out_size, kernel_size=p.KERNEL_SIZE, stride=1, padding='same')

        # self.conv = {}
        # for i in range(p.CASCADE_FILTERS):
        #     self.conv['CONV_' + str(i)] = nn.Conv2d(out_size, out_size, kernel_size=p.KERNEL_SIZE, stride=1, padding='same')

        self.conv = nn.ModuleDict()
        self.NB_CASCADE = p.CASCADE_FILTERS
        for i in range(p.CASCADE_FILTERS):
            self.conv['CONV_' + str(i)] = nn.Conv2d(out_size, out_size, kernel_size=p.KERNEL_SIZE, stride=1, padding='same')

        self.dropout = nn.Dropout2d(p.DROPOUT)
        self.instanceNorm = nn.InstanceNorm2d(out_size)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.conv2D(x)
        if self.normalization:
            x = self.instanceNorm(x)
        x = self.dropout(x)
        x = self.activation(x)

        for key in self.conv:
            x = self.conv[key](x)
            if self.normalization:
                x = self.instanceNorm(x)
            x = self.dropout(x)
            x = self.activation(x)

        x = self.MaxPool2d(x)

        return x

# ----------------------------------------------------------------------------------------------------------------------
class UNetUp(nn.Module):
    """Ascending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """

    def __init__(self, in_size, out_size, p):
        super(UNetUp, self).__init__()

        self.normalization = p.NORMALIZATION

        self.upsample = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)

        self.conv2d1 = nn.Conv2d(out_size, out_size, kernel_size=p.KERNEL_SIZE, stride=1, padding="same")

        self.conv = nn.ModuleDict()
        for i in range(p.CASCADE_FILTERS):
            self.conv['CONV_' + str(i)] = nn.Conv2d(out_size, out_size, kernel_size=p.KERNEL_SIZE, stride=1, padding="same")

        self.dropout = nn.Dropout2d(p.DROPOUT)
        self.normalization = nn.InstanceNorm2d(out_size)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, skip_input=None):

        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection

        x = self.upsample(x)
        if self.normalization:
            x = self.normalization(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.conv2d1(x)
        if self.normalization:
            x = self.normalization(x)
        x = self.dropout(x)
        x = self.activation(x)

        for key in self.conv.keys():
            x = self.conv[key](x)
            if self.normalization:
                x = self.normalization(x)
            x = self.dropout(x)
            x = self.activation(x)


        return x

# ----------------------------------------------------------------------------------------------------------------------
class FinalLayer(nn.Module):
    """Final block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size, p):

        super(FinalLayer, self).__init__()

        self.normalization = nn.InstanceNorm2d(out_size)

        # self.upsample = nn.Upsample(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(in_size, in_size, kernel_size=4, stride=2, padding=1)
        self.conv2d1 = nn.Conv2d(in_size, in_size, kernel_size=p.KERNEL_SIZE, padding="same")
        self.dropout = nn.Dropout2d(p.DROPOUT)

        self.conv_1 = nn.ModuleDict()
        for i in range(p.CASCADE_FILTERS):
            self.conv_1['CONV_' + str(i)] = nn.Conv2d(in_size, in_size, kernel_size=p.KERNEL_SIZE, stride=1, padding="same")
        self.droupout = nn.Dropout2d(p.DROPOUT)
        self.instanceNorm = nn.InstanceNorm2d(out_size)
        self.activation = nn.ReLU()
        self.conv2d2 = nn.Conv2d(in_size, out_size, kernel_size=p.KERNEL_SIZE, padding="same")

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x, skip_input=None):

        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection

        x = self.upsample(x)
        if self.normalization:
            x = self.instanceNorm(x)
        x = self.droupout(x)
        x = self.activation(x)

        x = self.conv2d1(x)
        if self.normalization:
            x = self.instanceNorm(x)
        x = self.droupout(x)
        x = self.activation(x)

        # --- cascade convolution
        for key in self.conv_1.keys():
            x = self.conv_1[key](x)
            if self.normalization:
                x = self.normalization(x)
            x = self.dropout(x)
            x = self.activation(x)

        x = self.conv2d2(x)

        return x

    # ------------------------------------------------------------------------------------------------------------------