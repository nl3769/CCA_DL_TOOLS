import torch

class lossClass():

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, p):

        self.criterion_GAN = torch.nn.BCEWithLogitsLoss()

        if p.LOSS == 'L1':
            self.criterion_pixelwise = lambda org, fake_org: self.compute_L1(org, fake_org)
        elif p.LOSS == 'L2':
            self.criterion_pixelwise = lambda org, fake_org: self.compute_L2(org, fake_org)
        elif p.LOSS == 'L1L2':
            self.criterion_pixelwise = lambda org, fake_org: self.compute_L1L2(org, fake_org)
        elif p.LOSS == 'histo_loss':
            self.criterion_pixelwise = lambda org, fake_org: self.compute_histo_loss(org, fake_org)
        
        self.lambda_GAN = p.lambda_GAN          # Weight generator loss
        self.lambda_pixel = p.lambda_pixel      # Weight discriminator loss

    # ------------------------------------------------------------------------------------------------------------------
    def compute_L1(self, org, fake_org):

        L1_metric = torch.nn.L1Loss(reduction='mean')

        return L1_metric(org, fake_org)

    # ------------------------------------------------------------------------------------------------------------------
    def compute_L2(self, org, fake_org):

        L2_metric = torch.nn.MSELoss(reduction='mean')

        return L2_metric(org, fake_org)

    # ------------------------------------------------------------------------------------------------------------------
    def compute_L1L2(self, org, fake_org):

        L1L2 = 0.5 * (self.compute_L1(org, fake_org) + self.compute_L2(org, fake_org))

        return L1L2

    # ------------------------------------------------------------------------------------------------------------------
    def compute_histo_loss(self, org, fake_org):
        org = torch.pow(org, 2)
        fake_org = torch.pow(fake_org, 2)
        
        L2 = self.compute_L2(org, fake_org)

        return L2

    # ------------------------------------------------------------------------------------------------------------------
    def compute_loss_generator(self, org, fake_org, discriminator_out, valid):

        loss_GAN = self.criterion_GAN(discriminator_out[0], valid)
        loss_pixel = self.criterion_pixelwise(fake_org, org)

        loss_generator = self.lambda_GAN * loss_GAN + self.lambda_pixel * loss_pixel

        return loss_generator, {'loss_GAN': loss_GAN, 'loss_pixel': loss_pixel}

    # ------------------------------------------------------------------------------------------------------------------
    def compute_loss_discriminator(self, discriminator_out, valid, fake):

        loss_real = self.criterion_GAN(discriminator_out[0], valid)
        loss_fake = self.criterion_GAN(discriminator_out[1], fake)

        loss_discriminator = 0.5 * (loss_real + loss_fake)

        return loss_discriminator

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, org, sim, fake_org, discriminator_out, valid, fake, generator = None):

        if generator == True:

            return self.compute_loss_generator(org, fake_org, discriminator_out, valid)

        elif generator == False:

            return self.compute_loss_discriminator(discriminator_out, valid, fake)
    # ------------------------------------------------------------------------------------------------------------------