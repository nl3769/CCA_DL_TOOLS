import torch
import math

import torch.nn                             as nn

# ----------------------------------------------------------------
class encoder(nn.Module):

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
    self.enc        = nn.ModuleDict({})
    self.nb_layers  = n_layers
    self.pool       = nn.AvgPool2d((2, 2), stride=(2, 2))

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

  # ---------------------------------------------------------------------------------------------------------------------
  def forward(self, x):

    skip = []

    for id, layer in enumerate(self.enc.keys()):
      x = self.enc[layer](x)
      if id < (self.nb_layers - 1):
        skip.append(x)
        x = self.pool(x)

    return x, skip

# ----------------------------------------------------------------
class decoder(nn.Module):

  def __init__(
    self,
    output_nc,
    n_layers,
    ngf,
    kernel_size,
    padding,
    use_bias,
    upconv,
    dropout=None
    ):

    super(decoder, self).__init__()
    self.dec      = nn.ModuleDict({})
    self.upscale = nn.ModuleDict({})
    self.nb_skip    = n_layers

    for i in range(n_layers-1, 0, -1):

        ch_out_ = ngf * 2 ** (i-1)
        ch_in_  = ngf * 2 ** i + ngf * 2 ** (i-1)
        
        if upconv == True:
            self.upscale["layers_" + str(i - 1)] = nn.ConvTranspose2d(ngf * 2 ** i, ngf * 2 ** i, kernel_size=4, stride=2, padding=1)
        else:
            self.upscale["layers_" + str(i - 1)] = nn.Upsample(scale_factor=2)

        if i == 1:
            self.dec["layers_" + str(i - 1)] = nn.Sequential(
                nn.Conv2d(ch_in_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                nn.BatchNorm2d(ch_out_),
                nn.PReLU(),
                nn.Conv2d(ch_out_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias,  padding_mode='replicate'),
                nn.BatchNorm2d(ch_out_),
                nn.PReLU(),
                nn.Conv2d(ch_out_, output_nc, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias,  padding_mode='replicate')
                )
        else:
            self.dec["layers_" + str(i - 1)] = nn.Sequential(
                nn.Conv2d(ch_in_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                nn.BatchNorm2d(ch_out_),
                nn.PReLU(),
                nn.Conv2d(ch_out_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                nn.BatchNorm2d(ch_out_),
                nn.PReLU()
                )

  # --------------------------------------------------------------------------------------------------------------------
  def forward(self, x, skip):

    for id, key in enumerate(self.dec.keys()):
        x = self.upscale[key](x)
        x = torch.cat((x, skip[self.nb_skip - id - 2]), 1)
        x = self.dec[key](x)

    return x

# -----------------------------------------------------------------------------------------------------------------------
class Unet(nn.Module):

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
        upconv,
        dropout             = None
        ):

        super(Unet, self).__init__()
        self.output_activation = output_activation
        # --- encoder part of the Unet
        self.enc = encoder(
          input_nc      = input_nc,
          n_layers      = n_layers,
          ngf           = ngf,
          kernel_size   = kernel_size,
          padding       = padding,
          use_bias      = use_bias
          )

        # --- decoder part of the Unet
        self.dec = decoder(
          output_nc         = output_nc,
          n_layers          = n_layers,
          ngf               = ngf,
          kernel_size       = kernel_size,
          padding           = padding,
          use_bias          = use_bias,
          upconv            = upconv
          )
          
        # ---------------------------------------------------------------------------------------------------------------------
    def forward(self, I):
    
    
        # --- Unet forward pass
        x, skip = self.enc(I)
        x       = self.dec(x, skip)
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x
