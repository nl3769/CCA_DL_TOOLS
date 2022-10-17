'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import package_loss.lossImplementation      as plli

# ----------------------------------------------------------------------------------------------------------------------
class lossSeg1Frame():

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self):
        """ TODO
        Args:
            TODO
        Returns:
            TODO
        """

        self.metrics = {'dice_I1': [],
                        'BCE_I1': []
                        }

        self.dice = plli.Dice()
        self.DiceBCELoss = plli.DiceBCELoss()
        self.BCE = plli.BCE()

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def seg_loss(pmask1, gtmask1, DiceBCELoss, dice, BCE):
        """ Loss function defined over pairs of images.
        Args:
            TODO
        Returns:
            TODO
        """

        # --- compute loss
        loss = DiceBCELoss(pmask1, gtmask1)
        # --- store metrics
        metrics = {
            'dice_I1': dice(pmask1, gtmask1),
            'BCE_I1': BCE(pmask1, gtmask1)
        }

        return loss, metrics

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, pmask1, gtmask1):
        """ TODO
        Args:
            TODO
        Returns:
            TODO
        """

        seg_loss, metrics = self.seg_loss(pmask1, gtmask1, self.DiceBCELoss, self.dice, self.BCE)

        return seg_loss, metrics

# ----------------------------------------------------------------------------------------------------------------------
