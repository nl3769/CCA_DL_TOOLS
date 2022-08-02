import numpy as np
import torch
import torch.nn as nn
import math

# ----------------------------------------------------------------------------------------------------------------------------------------------------
class encoder(nn.Module):

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        input_nc,
        n_layers,
        ngf,
        kernel_size,
        padding,
        use_bias,
        dropout=None
        ):

        super(encoder, self).__init__()
        self.enc = nn.ModuleDict({})
        self.nb_layers = n_layers
        self.pool     = nn.AvgPool2d((2, 2), stride=(2, 2))

        for i in range(n_layers):
            if i == 0:
                self.enc["layers_" + str(i + 1)] = nn.Sequential(
                    nn.Conv2d(input_nc, ngf, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                    nn.BatchNorm2d(ngf),
                    nn.PReLU(),
                    nn.Conv2d(ngf, ngf, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                    nn.BatchNorm2d(ngf),
                    nn.PReLU()
                    )
            else:
                ch_in_  = ngf * 2 ** (i-1)
                ch_out_ = ngf * 2 ** i
                self.enc["layers_" + str(i + 1)] = nn.Sequential(
                    nn.Conv2d(ch_in_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                    nn.BatchNorm2d(ch_out_),
                    nn.PReLU(),
                    nn.Conv2d(ch_out_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                    nn.BatchNorm2d(ch_out_),
                    nn.PReLU()
                    )

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def forward(self, x):

        skip = []
        for id, layer in enumerate(self.enc.keys()):
          x = self.enc[layer](x)
          if id < self.nb_layers:
            skip.append(x)
          x = self.pool(x)

        return x, skip

# ----------------------------------------------------------------------------------------------------------------------------------------------------
class decoder(nn.Module):

    def __init__(
        self,
        output_nc,
        n_layers,
        ngf,
        kernel_size,
        padding,
        use_bias,
        dropout=None
        ):

        super(decoder, self).__init__()
        self.dec = nn.ModuleDict({})
        self.upscale = nn.Upsample(scale_factor=2)
        self.nb_skip = n_layers - 1
        for i in range(n_layers, 0, -1):

            ch_out_ = ngf * 2 ** (i-1)
            ch_in_  = ngf * 2 ** (i) + ngf * 2 ** (i-1)

            if i == 1:
                self.dec["layers_" + str(i + 1)] = nn.Sequential(
                    nn.Conv2d(ch_in_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                    nn.BatchNorm2d(ch_out_),
                    nn.PReLU(),
                    nn.Conv2d(ch_out_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias,  padding_mode='replicate'),
                    nn.BatchNorm2d(ch_out_),
                    nn.PReLU(),
                    nn.Conv2d(ch_out_, output_nc, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias,  padding_mode='replicate'),
                    )
            else:
                self.dec["layers_" + str(i + 1)] = nn.Sequential(
                    nn.Conv2d(ch_in_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                    nn.BatchNorm2d(ch_out_),
                    nn.PReLU(),
                    nn.Conv2d(ch_out_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                    nn.BatchNorm2d(ch_out_),
                    nn.PReLU(),
                    nn.Conv2d(ch_out_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                    nn.BatchNorm2d(ch_out_),
                    nn.PReLU()
                    )

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def forward(self, x, skip):

        for id, key in enumerate(self.dec.keys()):
            x = self.upscale(x)
            x = torch.cat((x, skip[self.nb_skip-id]), 1)
            x = self.dec[key](x)

        return x

# ----------------------------------------------------------------------------------------------------------------------------------------------------
class bottleneck(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        dilatation_rate,
        n_layers,
        ngf,
        kernel_size,
        use_bias
        ):

        super(bottleneck, self).__init__()
        self.bottleneck = nn.ModuleDict({})
        self.nb_dil = len(dilatation_rate)
        input_nc = ngf * 2 ** (n_layers-1)

        for dil in dilatation_rate:
            padd = math.floor(kernel_size[0]/2) + dil - 1
            self.bottleneck['dilatation_' + str(dil)] = nn.Sequential(
                nn.Conv2d(input_nc, input_nc, kernel_size=kernel_size, stride=1, padding=padd, dilation=dil, bias=use_bias, padding_mode='replicate'),
                nn.BatchNorm2d(input_nc),
                nn.PReLU()
                )

        self.convb = nn.Conv2d(len(dilatation_rate) * input_nc, input_nc * 2, kernel_size=1, stride=1, padding=0, bias=use_bias, padding_mode='replicate')

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def forward(self, x):

        for id, key in enumerate(self.bottleneck.keys()):
            x = self.bottleneck[key](x)
            if id == 0:
                out = x
            else:
                out = torch.cat((x, out), 1)
        out = self.convb(out)

        return out

# ----------------------------------------------------------------------------------------------------------------------------------------------------
class dilatedUnet(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        input_nc,
        output_nc,
        n_layers,
        ngf,
        kernel_size,
        padding,
        use_bias,
        output_activation,
        dropout=None
        ):

        super(dilatedUnet, self).__init__()
        self.output_activation = output_activation
        self.enc = encoder(
            input_nc    = input_nc,
            n_layers    = n_layers,
            ngf         = ngf,
            kernel_size = kernel_size,
            padding     = padding,
            use_bias    = use_bias
            )

        self.dec = decoder(
            output_nc   = output_nc,
            n_layers    = n_layers,
            ngf         = ngf,
            kernel_size = kernel_size,
            padding     = padding,
            use_bias    = use_bias
            )

        self.bottleneck = bottleneck(
            dilatation_rate = [1, 2, 3, 4],
            n_layers        = n_layers,
            ngf             = ngf,
            kernel_size     = (3, 3),
            use_bias        = use_bias
            )
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def forward(self, x):

        x, skip = self.enc(x)
        x       = self.bottleneck(x)
        x       = self.dec(x, skip)
        x = self.output_activation(x)
        
        return x
