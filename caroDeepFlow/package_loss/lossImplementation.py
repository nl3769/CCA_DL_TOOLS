import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------------
class Dice(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, inputs, targets, smooth=1):

        # --- flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice

# ----------------------------------------------------------------------------------------------------------------------
class DiceBCELoss(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, inputs, targets, smooth=1):

        # --- flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

# ----------------------------------------------------------------------------------------------------------------------
class BCE(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, weight=None, size_average=True):
        super(BCE, self).__init__()

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, inputs, targets, smooth=1):

        # --- flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return BCE

# ----------------------------------------------------------------------------------------------------------------------