'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import package_network.update               as  updateNet
import package_network.features_extractor   as  featureExtractor
import package_network.correlation          as  correlation
import package_network.grid_handler         as  gridHandler
import package_network.decoder              as  pnd
import torch
import torch.nn                             as  nn
import torch.nn.functional                  as  F


# ----------------------------------------------------------------------------------------------------------------------
class NetSegDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p):
        super(NetSegDecoder, self).__init__()

        self.conv1 = nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv2 = nn.Conv2d(640, 128, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv4 = nn.Conv2d(64, 2, kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.upconv1 = nn.ConvTranspose2d(384, 128, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.upconv2 = nn.ConvTranspose2d(320, 128, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=1)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.batchNorm1 = nn.BatchNorm2d(128, affine=False)
        self.batchNorm2 = nn.BatchNorm2d(128, affine=False)
        self.batchNorm3 = nn.BatchNorm2d(128, affine=False)
        self.batchNorm4 = nn.BatchNorm2d(128, affine=False)
        self.batchNorm5 = nn.BatchNorm2d(128, affine=False)
        self.batchNorm6 = nn.BatchNorm2d(64, affine=False)

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

# ----------------------------------------------------------------------------------------------------------------------
class NetEncoder(nn.Module):
    """Encoder part of the network. Features map are joied for segmentation and flow estimation. """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p):
        super(NetEncoder, self).__init__()
        self.p = p
        self.fnet1 = featureExtractor.BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.p.DROPOUT)
        self.fnet2 = featureExtractor.BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.p.DROPOUT)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, I1, I2):

        I1 = I1.contiguous()
        I2 = I2.contiguous()

        # --- run the feature network
        fmap1, skc1 = self.fnet1(I1)
        fmap2, skc2 = self.fnet2(I2)

        return fmap1, skc1, fmap2, skc2

# ----------------------------------------------------------------------------------------------------------------------
class NetFlow(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p):
        super(NetFlow, self).__init__()
        self.p = p

        # --- we use the biggest model
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        # --- features extraction
        self.cnet = featureExtractor.BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=self.p.DROPOUT)
        # --- update block
        self.update_block = updateNet.BasicUpdateBlock(self.p, hidden_dim=hdim)

    # ------------------------------------------------------------------------------------------------------------------
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0. """

        N, C, H, W = img.shape
        coords0 = gridHandler.coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = gridHandler.coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    # ------------------------------------------------------------------------------------------------------------------
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination. """

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_flow = up_flow.reshape(N, 2, 8 * H, 8 * W)

        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, I1, fmap1, fmap2, M1, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pairs of images. """


        I1 = I1.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.p.ALTERNATE_COORDINATE:
            corr_fn = correlation.AlternateCorrBlock(fmap1, fmap2, radius=self.p.CORRELATION_RADIUS)
        else:
            corr_fn = correlation.CorrBlock(fmap1, fmap2, radius=self.p.CORRELATION_RADIUS)

        # --- run the context network
        cnet, _ = self.cnet(I1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(I1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)         # index correlation volume (on enregistre la correlation interpolee à tous les niveaux de la pyramide)

            flow = coords1 - coords0        # on calcul le flow
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow

            # --- Upsample prediction
            if up_mask is None:
                flow_up = gridHandler.upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        for id_pred in range(len(flow_predictions)):
            flow_predictions[id_pred] = torch.mul(flow_predictions[id_pred], M1)

        return flow_predictions

    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------