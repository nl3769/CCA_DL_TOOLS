import torch

class lossFlow():

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self):

        self.metrics = {'epe': [],
                        '1px': [],
                        '3px': [],
                        '5px': []}

    # ------------------------------------------------------------------------------------------------------------------

    def sequence_loss(self, flow_preds: torch.Tensor, flow_gt: torch.Tensor, gamma: float):
        """ Loss function defined over sequence of flow predictions. """

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # --- exclude invalid pixels and extremely large diplacements
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += (i_weight * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item()}

        return flow_loss, metrics

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, flow_preds: torch.Tensor, flow_gt: torch.Tensor, gamma: float):

        flow_loss, metrics = self.sequence_loss(flow_preds, flow_gt, gamma)

        return flow_loss, metrics

    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------