import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
from torchvision import transforms


class GeneratorSR(nn.Module):

    def __init__(self, p, in_channels=1, out_channels=1):
        super(GeneratorSR, self).__init__()

        self.nb_filters = 64
        self.conv1 = nn.Conv2d(in_channels, self.nb_filters, kernel_size=9, stride=1, padding='same')
        self.relu = nn.ReLU()

        self.conv_res, self.batchnorm_res = self.init_residual_block(self.nb_filters)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, kernel_size=3, stride=1, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(self.nb_filters)
        self.conv3 = nn.Conv2d(self.nb_filters, 256, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(256, out_channels, kernel_size=9, stride=1, padding='same')

    # ------------------------------------------------------------------------------------------------------------------
    def residual_block(self, x, n_step):

        for step in range(n_step):
            x_0 = x
            str_step = str(step) + '_0'
            x = self.conv_res[str_step](x)
            x = self.batchnorm_res[str_step](x)
            x = self.relu(x)

            str_step = str(step) + '_1'
            x = self.conv_res[str_step](x)
            x = self.batchnorm_res[str_step](x)

            x = torch.add(x, x_0)

        return x

    # ------------------------------------------------------------------------------------------------------------------
    def init_residual_block(self, nb_filters, n_step=5):

        conv_res = nn.ModuleDict()
        batchnorm_res = nn.ModuleDict()

        for step in range(n_step):
            str_step = str(step) + '_0'
            conv_res[str_step] = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, stride=1, padding='same')
            batchnorm_res[str_step] = nn.BatchNorm2d(nb_filters)

            str_step = str(step) + '_1'
            conv_res[str_step] = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, stride=1, padding='same')
            batchnorm_res[str_step] = nn.BatchNorm2d(nb_filters)

        return conv_res, batchnorm_res

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)

        x1 = self.residual_block(x, 5)

        x1 = self.conv2(x1)
        x1 = self.batchnorm2(x1)
        x1 = torch.add(x, x1)

        x1 = self.conv3(x1)
        x1 = self.relu(x1)

        x1 = self.conv4(x1)
        x1 = self.relu(x1)

        x1 = self.conv5(x1)

        return x1

# ----------------------------------------------------------------------------------------------------------------------
class DiscriminatorSR(nn.Module):

    def __init__(self, p, in_channels=1, out_channels=1):
        super(DiscriminatorSR, self).__init__()

        self.conv1 = nn.Conv2d(2*in_channels, 64, kernel_size=3, stride=1, padding='same')
        self.leakyRelu = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same')
        self.batchnorm5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same')
        self.batchnorm7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 32, kernel_size=3, stride=2, padding=1)
        self.batchnorm8 = nn.BatchNorm2d(32)

        self.linear_layer1 = nn.Linear(16384, 1024)
        self.linear_layer2 = nn.Linear(1024, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):

        x = torch.concat((x1, x2), 1)

        x = self.conv1(x)
        x = self.leakyRelu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leakyRelu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.leakyRelu(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.leakyRelu(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.leakyRelu(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.leakyRelu(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.leakyRelu(x)

        x = self.conv8(x)
        x = self.batchnorm8(x)
        x = self.leakyRelu(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = x.unsqueeze(dim=1).unsqueeze(dim=1)
        x = self.linear_layer1(x)
        x = self.leakyRelu(x)
        x = self.linear_layer2(x)
        x = self.sigmoid(x)

        return x

