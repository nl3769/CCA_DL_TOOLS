'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import torch
import torch.nn                     as nn

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def residual_block_f(
    in_ch,
    out_ch,
    activation,
    stride,
    kernel_size,
    padding,
    use_bias
    ):
    
    res_block = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, padding_mode='replicate'),
        activation
        )
    
    return res_block

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def residual_block(
    in_ch,
    n_filters,
    activation,
    stride,
    kernel_size,
    padding,
    use_bias
    ):

    res_block = nn.Sequential(
        nn.Conv2d(in_ch, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, padding_mode='replicate'),
        nn.BatchNorm2d(n_filters),
        activation,
        nn.Conv2d(in_ch, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, padding_mode='replicate'),
        nn.BatchNorm2d(n_filters),
        activation
        )
    
    return res_block

# ----------------------------------------------------------------------------------------------------------------------------------------------------
class SRGan(nn.Module):
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        input_nc,
        output_nc,
        kernel_size,
        padding,
        use_bias,
        dropout=None
        ):

        super(SRGan, self).__init__()
        res_nb           = 3
        res_nb_f         = 2
        nb_filters       = 64
        stride           = 1
        self.res_block   = nn.ModuleDict({})
        self.res_block_f = nn.ModuleDict({})
        
        # --- first layer of the network
        self.input_block = nn.Sequential(
            nn.Conv2d(input_nc, nb_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, padding_mode='replicate'),
            nn.PReLU()
            )

        # --- residual blocks
        for i in range(res_nb):
            self.res_block["res_block_" + str(i)] = residual_block(nb_filters, nb_filters, nn.PReLU(), stride, kernel_size, padding, use_bias)
        
        # --- "gather" results of residual blocks
        self.gather = nn.Sequential(
            nn.Conv2d(nb_filters, nb_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, padding_mode='replicate'),
            nn.BatchNorm2d(nb_filters)
            )

        # --- reduce filters dimension 
        in_ch = nb_filters
        for i in range(res_nb_f):
            out_ch = int(in_ch / 2)
            self.res_block_f["res_block_f_" + str(i)] = residual_block_f(in_ch, out_ch, nn.PReLU(), stride, kernel_size, padding, use_bias)
            in_ch = out_ch

        # --- final convolution
        self.conv_out = nn.Conv2d(in_ch, output_nc, kernel_size=(9, 9), stride=stride, padding=(4, 4), bias=use_bias, padding_mode='replicate')
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    def forward(self, I):
        
        skip_connection = []
        I = self.input_block(I)
        skip_connection.append(I)

        # --- loop over residual blocks
        for idx, key in enumerate(self.res_block):
            I = self.res_block[key](I)
            I = torch.add(I, skip_connection[idx])
            skip_connection.append(I)
        
        I = self.gather(I)
        
        # --- add skip connection
        I = torch.add(I, skip_connection[0])

        # --- reduce filters dimension
        for idx, key in enumerate(self.res_block_f):
            I = self.res_block_f[key](I)

        I = self.conv_out(I)

        return I
