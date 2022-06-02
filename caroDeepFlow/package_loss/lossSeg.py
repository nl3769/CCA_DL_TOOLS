'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import package_loss.lossImplementation      as plli

class lossSeg():

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self):

        self.metrics = {'dice_I1': [],
                        'dice_I2': [],
                        'BCE_I1': [],
                        'BCE_I2': []}

        self.dice = plli.Dice()
        self.DiceBCELoss = plli.DiceBCELoss()
        self.BCE = plli.BCE()

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def seg_loss(pmask1, pmask2, gtmask1, gtmask2, DiceBCELoss, dice, BCE):
        """ Loss function defined over pairs of images. """

        # --- compute loss
        DiceBCE1 = DiceBCELoss(pmask1, gtmask1)
        DiceBCE2 = DiceBCELoss(pmask2, gtmask2)
        loss = (DiceBCE1 + DiceBCE2) / 2

        # --- compute dice
        diceI1 = dice(pmask1, gtmask1)
        diceI2 = dice(pmask2, gtmask2)

        # --- compute BCE
        BCEI1 = BCE(pmask1, gtmask1)
        BCEI2 = BCE(pmask2, gtmask2)

        metrics = {'dice_I1': diceI1,
                   'dice_I2': diceI2,
                   'BCE_I1': BCEI1,
                   'BCE_I2': BCEI2}

        return loss, metrics

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, pmask1, pmask2, gtmask1, gtmask2):

        seg_loss, metrics = self.seg_loss(pmask1, pmask2, gtmask1, gtmask2, self.DiceBCELoss, self.dice, self.BCE)

        return seg_loss, metrics

# ----------------------------------------------------------------------------------------------------------------------