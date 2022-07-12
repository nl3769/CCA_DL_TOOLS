
def display_loss(validation: bool, epoch: int, loss_training: float, loss_org_train: float,  loss_val: float, loss_org_val: float):
    """ Display loss at the end of each epoch. """

    # --- display loss
    if validation:
        loss_GAN_train = loss_org_train['loss_GAN']
        loss_pixel_train = loss_org_train['loss_pixel']
        loss_GAN_val = loss_org_val['loss_GAN']
        loss_pixel_val = loss_org_val['loss_pixel']

        print(f'EPOCH {epoch} --- training loss: {loss_training}  / loss_GAN: {loss_GAN_train} / loss_pixel: {loss_pixel_train}')
        print(f'EPOCH {epoch} --- validation loss: {loss_val}  / loss_GAN: {loss_GAN_val} / loss_pixel: {loss_pixel_val}')
    else:
        loss_GAN_train = loss_org_train['loss_GAN']
        loss_pixel_train = loss_org_train['loss_pixel']
        print(f'EPOCH {epoch} --- training loss: {loss_training}  / loss_GAN: {loss_GAN_train} / loss_pixel: {loss_pixel_train}')

# ----------------------------------------------------------------------------------------------------------------------