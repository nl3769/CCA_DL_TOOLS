'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import torch
import torch.nn                             as  nn


# ----------------------------------------------------------------------------------------------------------------------
class NetSegDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p):
        super(NetSegDecoder, self).__init__()

        self.conv1 = nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv2 = nn.Conv2d(640, 128, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv3 = nn.Conv2d(128, 64,  kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv4 = nn.Conv2d(64, 2,    kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.upconv1 = nn.ConvTranspose2d(384, 128, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.upconv2 = nn.ConvTranspose2d(320, 128, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64,  kernel_size=(4, 4), stride=(2, 2), padding=1)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.batchNorm1 = nn.BatchNorm2d(128, affine=True)
        self.batchNorm2 = nn.BatchNorm2d(128, affine=True)
        self.batchNorm3 = nn.BatchNorm2d(128, affine=True)
        self.batchNorm4 = nn.BatchNorm2d(128, affine=True)
        self.batchNorm5 = nn.BatchNorm2d(128, affine=True)
        self.batchNorm6 = nn.BatchNorm2d(64,  affine=True)

        self.activation = nn.Sigmoid()

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, fmap1, skc1, fmap2, skc2):

        bottleneck = torch.cat((fmap1, fmap2), dim=1)
        x = self.conv1(bottleneck)
        x = self.relu1(x)
        x = self.batchNorm1(x)

        x = torch.cat((x, skc1[4], skc2[4]), dim=1)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchNorm2(x)

        x = torch.cat((x, skc1[3], skc2[3]), dim=1)
        x = self.upconv1(x)
        x = self.relu3(x)
        x = self.batchNorm3(x)

        x = torch.cat((x, skc1[2], skc2[2]), dim=1)
        x = self.upconv2(x)
        x = self.relu4(x)
        x = self.batchNorm4(x)

        x = torch.cat((x, skc1[1], skc2[1]), dim=1)
        x = self.upconv3(x)
        x = self.relu5(x)
        x = self.batchNorm5(x)

        x = self.conv3(x)
        x = self.relu6(x)
        x = self.batchNorm6(x)

        x = self.conv4(x)

        x = self.activation(x)

        M1, M2 = torch.chunk(x, 2, 1)

        return M1, M2
