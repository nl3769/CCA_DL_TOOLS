import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------------
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(f'{x.shape = }')
        x = self.conv1(x)
        # print(f'{x.shape = }')
        x = self.relu(x)
        # print(f'{x.shape = }')
        x = self.conv2(x)
        # print(f'{x.shape = }')
        return x

# ----------------------------------------------------------------------------------------------------------------------
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q

        return h

# ----------------------------------------------------------------------------------------------------------------------
class SepConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()


        self.convz1 = nn.Conv2d( hidden_dim + input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5,1), padding=(2,0))

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, h, x):

        # --- horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # --- vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q


        return h
# ----------------------------------------------------------------------------------------------------------------------

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

# ----------------------------------------------------------------------------------------------------------------------
class BasicMotionEncoder(nn.Module):
    def __init__(self, p):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = p.CORRELATION_LEVEL * (2*p.CORRELATION_RADIUS + 1)**2

        # k = 2 if '3T' in p.MODEL_NAME else 1
        k = 1
        self.convc1 = nn.Conv2d(cor_planes * k, 256 * k, 1, padding=0)
        self.convc2 = nn.Conv2d(256 * k, 192 * k, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192 * k, (128-2) * k, 3, padding=1)

    def forward(self, flow, corr):

        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

# ----------------------------------------------------------------------------------------------------------------------
class SmallUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()

        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):

        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

# ----------------------------------------------------------------------------------------------------------------------
class BasicUpdateBlock(nn.Module):

    def __init__(self, p, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()

        self.p = p
        self.encoder = BasicMotionEncoder(p)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 64 * 9, 1, padding=0))  # 64 = 8*8 -> et on multiplie par 9 pour otbenir 9 coefficient pour chaque pixel

    def forward(self, net, inp, corr, flow):

        """
        :param net: output vector of GRU (down-sampled flow)
        :param inp: output of context encoder
        :param corr: concatenated pyramid correlation
        :param flow: current flow (coords1 - coords0), coords0 is fixed but coords1 is updated at each iteration
        :return: net, mask, delta_flow (upsample)
        """

        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = self.mask(net)
        mask = 0.25 * mask # scale mask to balance gradients

        return net, mask, delta_flow

# ----------------------------------------------------------------------------------------------------------------------
class BasicUpdateBlock_3T(nn.Module):

    def __init__(self, p, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock_3T, self).__init__()

        self.p = p
        self.encoder = BasicMotionEncoder(p)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=127 * 2 + hidden_dim)

        self.flow_head_01 = FlowHead(hidden_dim, hidden_dim=256)
        self.flow_head_12 = FlowHead(hidden_dim, hidden_dim=256)

        self.hidden_dim = hidden_dim
        self.mask_01 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 64 * 9, 1, padding=0))  # 64 = 8*8 -> et on multiplie par 9 pour otbenir 9 coefficients pour chaque pixel

        self.mask_12 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 64 * 9, 1, padding=0))  # 64 = 8*8 -> et on multiplie par 9 pour otbenir 9 coefficients pour chaque pixel

    def forward(self, net, inp, corr, flow):
        """
        :param net: output vector of GRU (down-sampled flow)
        :param inp: output of context encoder
        :param corr: concatenated pyramid correlation
        :param flow: current flow (coords1 - coords0), coords0 is fixed but coords1 is updated at each iteration
        :return: net, mask, delta_flow (upsample)
        """

        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)

        # --- get delta flow
        delta_flow_01 = self.flow_head_01(net)
        delta_flow_12 = self.flow_head_01(net)

        # --- mask to upsample images
        mask_01 = self.mask_01(net)
        mask_01 = 0.25 * mask_01

        mask_12 = self.mask_12(net)
        mask_12 = 0.25 * mask_12  # scale mask to balance gradients

        delta_flow = {"delta_flow_01": delta_flow_01,
                      "delta_flow_12": delta_flow_12}
        mask = {"maks_01": mask_01,
                "maks_12": mask_12}

        return net, mask, delta_flow

# ----------------------------------------------------------------------------------------------------------------------
