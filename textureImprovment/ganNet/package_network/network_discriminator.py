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
        self.disc1 = discriminator_block(in_channels*2, 64)
        self.disc2 = discriminator_block(64, 128)
        self.disc3 = discriminator_block(128, 256)
        self.disc4 = discriminator_block(256, 512)
        self.conv = nn.Conv2d(512, 1, 1, stride=(2, 1), padding=0)
        # self.layers = nn.Sequential(
        #     discriminator_block(in_channels*2, 64),
        #     discriminator_block(64, 128),
        #     discriminator_block(128, 256),
        #     discriminator_block(256, 512),
        #     nn.Conv2d(512, 1, 1, stride=(2, 1), padding=0),
        # )


    def forward(self, img_A, img_B):

        # --- Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)

        # output = self.layers(img_input)

        output = self.disc1(img_input)
        output = self.disc2(output)
        output = self.disc3(output)
        output = self.disc4(output)
        output = self.conv(output)

        return output