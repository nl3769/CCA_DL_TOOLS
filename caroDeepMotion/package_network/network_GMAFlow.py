'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import torch

import package_network.network_featuresExtractor    as featureExtractor
import package_network.correlation                  as correlation
import package_network.grid_handler                 as gridHandler
import torch.nn                                     as nn
import torch.nn.functional                          as F

from package_network.network_gmaUpdator             import GMAUpdateBlock
from package_network.network_gma                    import Attention

# ----------------------------------------------------------------------------------------------------------------------
class GMA_NetFlow(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p):
        super(GMA_NetFlow, self).__init__()
        self.p = p

        # --- use the biggest model
        self.hidden_dim  = p.HIDDEN_DIM
        self.context_dim = p.CONTEXT_DIM
        self.num_heads   = p.NUM_HEAD

        # --- features extraction
        self.cnet = featureExtractor.BasicEncoder(output_dim=self.hidden_dim + self.context_dim, norm_fn='batch', dropout=self.p.DROPOUT)
        
        # --- update block
        self.update_block = GMAUpdateBlock(p, self.hidden_dim, self.num_heads)

        # --- attention module
        self.att = Attention(dim=self.context_dim, heads=self.num_heads, max_pos_size=160, dim_head=self.context_dim)

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
        """ Estimate displacement field between pair of images. """

        I1 = I1.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.p.ALTERNATE_COORDINATE:
            corr_fn = correlation.AlternateCorrBlock(fmap1, fmap2, radius=self.p.CORRELATION_RADIUS)
        else:
            corr_fn = correlation.CorrBlock(fmap1, fmap2, radius=self.p.CORRELATION_RADIUS)

        # --- context network
        cnet, _             = self.cnet(I1)
        net, inp            = torch.split(cnet, [hdim, cdim], dim=1)
        net                 = torch.tanh(net)
        inp                 = torch.relu(inp)
        attention           = self.att(inp)
        coords0, coords1    = self.initialize_flow(I1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)         # index correlation volume (on enregistre la correlation interpolee à tous les niveaux de la pyramide)
            
            # --- residual flow
            flow = coords1 - coords0
           
            # --- update flow
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)
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
