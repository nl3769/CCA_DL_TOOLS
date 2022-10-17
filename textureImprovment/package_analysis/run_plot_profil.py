import os

import matplotlib.pyplot        as plt
import numpy                    as np

from PIL                        import Image

if __name__ == '__main__':
    
    pinvivo     = "/home/laine/pc/Desktop/GAN_inference/img_org/tech_106.png"
    pinsilico   = "/home/laine/pc/Desktop/GAN_inference/img_sim/tech_106.png"
    pgan        = "/home/laine/pc/Desktop/GAN_inference/img_gan/tech_106.png"
    pres        = "/home/laine/Desktop/gan_inference"

    invivo      = np.array(Image.open(pinvivo))
    insilico    = np.array(Image.open(pinsilico))
    gan         = np.array(Image.open(pgan))

    height, width = invivo.shape

    vertical_profil_id      = int(height/1.5)
    horizontal_profil_id    = int(width / 2)

    vertical_profil_invivo = invivo[vertical_profil_id,]
    horizontal_profil_invivo = invivo[:, horizontal_profil_id]
    invivo = np.repeat(invivo[:, :, np.newaxis], 3, axis=2)
    invivo[vertical_profil_id,] = [0, 255, 0]
    invivo[:, horizontal_profil_id, :] = [0, 255, 0]

    vertical_profil_insilico = insilico[vertical_profil_id,]
    horizontal_profil_insilico = insilico[:, horizontal_profil_id]
    insilico = np.repeat(insilico[:, :, np.newaxis], 3, axis=2)
    insilico[vertical_profil_id,] = [255, 0, 0]
    insilico[:, horizontal_profil_id, :] = [255, 0, 0]

    vertical_profil_gan = gan[vertical_profil_id,]
    horizontal_profil_gan = gan[:, horizontal_profil_id]
    gan = np.repeat(gan[:, :, np.newaxis], 3, axis=2)
    gan[vertical_profil_id,] = [255, 120, 0]
    gan[:, horizontal_profil_id, :] = [255, 120, 0]

    plt.figure()
    plt.plot(vertical_profil_gan, color = 'tab:orange', label='GAN')
    plt.plot(vertical_profil_invivo, color='tab:green', label='in vivo')
    plt.plot(vertical_profil_insilico, color='tab:red', label='in silico')
    plt.legend()
    plt.savefig(os.path.join(pres, "horizontal_profil.svg"))

    plt.figure()
    plt.plot(horizontal_profil_gan, color = 'tab:orange', label='GAN')
    plt.plot(horizontal_profil_invivo, color='tab:green', label='in vivo')
    plt.plot(horizontal_profil_insilico, color='tab:red', label='in silico')
    plt.legend()
    plt.savefig(os.path.join(pres, "vertical_profil.svg"))

    plt.figure()
    plt.imshow(invivo)
    plt.savefig(os.path.join(pres, "invivo.svg"))

    plt.figure()
    plt.imshow(insilico)
    plt.savefig(os.path.join(pres, "insilico.svg"))

    plt.figure()
    plt.imshow(gan)
    plt.savefig(os.path.join(pres, "gan.svg"))

    plt.show()
