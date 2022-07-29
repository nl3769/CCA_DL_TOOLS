import torch
import math

import torch.nn                             as nn
import numpy                                as np

from package_network.positionalEmbedding    import SinusoidalPositionEmbeddings

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
    time_emb_dim,
    dropout=None):

    super(encoder, self).__init__()
    self.enc_0      = nn.ModuleDict({})
    self.enc_1      = nn.ModuleDict({})
    self.time_mlp   = nn.ModuleDict({})
    self.nb_layers  = n_layers
    self.pool       = nn.AvgPool2d((2, 2), stride=(2, 2))

    for i in range(n_layers + 1):
      if i == 0:
        self.enc_0["layers_" + str(i + 1)] = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(negative_slope=0.1))

        self.time_mlp["layers_" + str(i + 1)] = nn.Sequential(
            nn.Linear(time_emb_dim, ngf),
            nn.LeakyReLU(negative_slope=0.1)
            )

        self.enc_1["layers_" + str(i + 1)] = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(negative_slope=0.1)
            )


      else:
        ch_in_  = ngf * 2 ** (i-1)
        ch_out_ = ngf * 2 ** i
        self.enc_0["layers_" + str(i + 1)] = nn.Sequential(
            nn.Conv2d(ch_in_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
            nn.BatchNorm2d(ch_out_),
            nn.LeakyReLU(negative_slope=0.1)
            )

        self.time_mlp["layers_" + str(i + 1)] = nn.Sequential(
            nn.Linear(time_emb_dim, ch_out_),
            nn.LeakyReLU(negative_slope=0.1)
            )

        self.enc_1["layers_" + str(i + 1)] = nn.Sequential(
            nn.Conv2d(ch_out_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
            nn.BatchNorm2d(ch_out_),
            nn.LeakyReLU(negative_slope=0.1)
            )
    
    ch_in_  = ngf * 2 ** (n_layers)
    ch_out_ = ngf * 2 ** (n_layers + 1)
    '''
    self.encod_final_0 = nn.Sequential(
        nn.Conv2d(ch_in_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
        nn.BatchNorm2d(ch_out_),
        nn.LeakyReLU(negative_slope=0.1)
    )
    self.time_mlp["layers_" + str(i + 1)] = nn.Sequential(
        nn.Linear(time_emb_dim, ch_out_),
        nn.LeakyReLU(negative_slope=0.1)
        )
    self.encod_final_1 = nn.Sequential(
        nn.Conv2d(ch_ou_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
        nn.BatchNorm2d(ch_out_),
        nn.LeakyReLU(negative_slope=0.1)
    )
    '''

  # ---------------------------------------------------------------------------------------------------------------------
  def forward(self, x, t):

    skip = []

    for id, layer in enumerate(self.enc_0.keys()):
      x = self.enc_0[layer](x)
      time_emb = self.time_mlp[layer](t)
      time_emb = time_emb[(..., ) + (None, ) * 2]
      x = x + time_emb
      x = self.enc_1[layer](x)
      if id < self.nb_layers:
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
    time_emb_dim,
    dropout=None):

    super(decoder, self).__init__()
    self.dec_0      = nn.ModuleDict({})
    self.dec_1      = nn.ModuleDict({})
    self.time_mlp   = nn.ModuleDict({})
    self.upscale    = nn.Upsample(scale_factor=2)
    self.nb_skip    = n_layers - 1

    for i in range(n_layers, 0, -1):

        ch_out_ = ngf * 2 ** (i-1)
        ch_in_  = ngf * 2 ** (i) + ngf * 2 ** (i-1)

        if i == 1:
            self.dec_0["layers_" + str(i + 1)] = nn.Sequential(
                nn.Conv2d(ch_in_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                nn.BatchNorm2d(ch_out_),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(ch_out_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias,  padding_mode='replicate'),
                nn.BatchNorm2d(ch_out_),
                nn.LeakyReLU(negative_slope=0.1)
                )

            self.time_mlp["layers_" + str(i + 1)] = nn.Sequential(
                nn.Linear(time_emb_dim, ch_out_),
                nn.LeakyReLU(negative_slope=0.1)
                )

            self.dec_1["layers_" + str(i + 1)] = nn.Sequential(
                nn.Conv2d(ch_out_, output_nc, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias,  padding_mode='replicate'),
                )
        else:
            self.dec_0["layers_" + str(i + 1)] = nn.Sequential(
                nn.Conv2d(ch_in_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                nn.BatchNorm2d(ch_out_),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(ch_out_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                nn.BatchNorm2d(ch_out_),
                nn.LeakyReLU(negative_slope=0.1)
                )

            self.time_mlp["layers_" + str(i + 1)] = nn.Sequential(
                nn.Linear(time_emb_dim, ch_out_),
                nn.LeakyReLU(negative_slope=0.1)
                )

            self.dec_1["layers_" + str(i + 1)] = nn.Sequential(
                nn.Conv2d(ch_out_, ch_out_, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate'),
                nn.BatchNorm2d(ch_out_),
                nn.LeakyReLU(negative_slope=0.1)
                )

  # --------------------------------------------------------------------------------------------------------------------
  def forward(self, x, skip, t):

    for id, key in enumerate(self.dec_0.keys()):
        x = self.upscale(x)
        x = torch.cat((x, skip[self.nb_skip-id]), 1)
        x = self.dec_0[key](x)
        time_emb = self.time_mlp[key](t)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        x = x + time_emb
        x = self.dec_1[key](x)

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
        time_emb_dim,
        dropout      = None):

        super(Unet, self).__init__()
    
        # --- time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU())

        # --- encoder part of the Unet
        self.enc = encoder(
          input_nc      = input_nc,
          n_layers      = n_layers,
          ngf           = ngf,
          kernel_size   = kernel_size,
          padding       = padding,
          use_bias      = use_bias,
          time_emb_dim  = time_emb_dim
          )

        # --- decoder part of the Unet
        self.dec = decoder(
          output_nc     = output_nc,
          n_layers      = n_layers,
          ngf           = ngf,
          kernel_size   = kernel_size,
          padding       = padding,
          use_bias      = use_bias,
          time_emb_dim  = time_emb_dim
          )
          
        # ---------------------------------------------------------------------------------------------------------------------
    def forward(self, I, time_step):
    
        # --- embedd time
        t = self.time_mlp(time_step)
    
        # --- Unet forward pass
        x, skip = self.enc(I, t)
        x       = self.dec(x, skip, t)

        return x
