'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import package_network.features_extractor   as  featureExtractor
import torch.nn                             as  nn

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

        # --- package_core the feature network
        fmap1, skc1 = self.fnet1(I1)
        fmap2, skc2 = self.fnet2(I2)

        return fmap1, skc1, fmap2, skc2
