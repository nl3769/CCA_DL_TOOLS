import pytorch_lightning as pl

class LitAutoEncoder(pl.LightningModule):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, network, param, loss):

        self.network = network
        self.param = param
        self.loss = loss

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, img):  # noqa: D102
        return self.net(img)

    # ------------------------------------------------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        I, M, CF, fname = batch

        pred = self.networks(I)
        loss_val, metrics_val = self.loss(pred, M)

    # # ------------------------------------------------------------------------------------------------------------------
    # def configure_optimizers(self):
    #     a=1

