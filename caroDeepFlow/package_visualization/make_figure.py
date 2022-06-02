import matplotlib.pyplot as plt
import numpy as np

def make_figure(I1, I2, M1, M2, xOF, zOF, argOF, normOF, pres):

    ftsize = 6

    I1 = np.array(I1).astype(np.int16)
    I2 = np.array(I2).astype(np.int16)
    M1 = np.array(M1).astype(np.int16)
    M2 = np.array(M2).astype(np.int16)

    plt.rcParams['text.usetex'] = True
    plt.figure()

    # ------ IMAGE
    plt.subplot2grid((4, 6), (0, 0), colspan=1, rowspan=1)
    plt.imshow(I1, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'I1', fontsize=ftsize)
    # ------------
    plt.subplot2grid((4, 6), (0, 1), colspan=1, rowspan=1)
    plt.imshow(I2, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'I2', fontsize=ftsize)
    # ------------
    plt.subplot2grid((4, 6), (0, 2), colspan=1, rowspan=1)
    plt.imshow(I2-I1, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'I2 - I1', fontsize=ftsize)

    # ------ MASK
    plt.subplot2grid((4, 6), (1, 0), colspan=1, rowspan=1)
    plt.imshow(M1, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'M1', fontsize=ftsize)
    # ------------
    plt.subplot2grid((4, 6), (1, 1), colspan=1, rowspan=1)
    plt.imshow(M2, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'M2', fontsize=ftsize)
    # ------------
    plt.subplot2grid((4, 6), (1, 2), colspan=1, rowspan=1)
    plt.imshow(np.abs(M2-M1), cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'$|M2 - M1|$', fontsize=ftsize)

    # ------ OPTICAL FLOW
    plt.subplot2grid((4, 6), (2, 0), colspan=1, rowspan=1)
    plt.imshow(normOF, cmap='hot')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'$\sqrt{x_{OF_{1\rightarrow 2}}^2 + z_{OF_{1\rightarrow 2}}^2}$', fontsize=ftsize)
    # ------------
    plt.subplot2grid((4, 6), (2, 1), colspan=1, rowspan=1)
    plt.imshow(xOF, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'$X displacement$', fontsize=ftsize)
    # ------------
    plt.subplot2grid((4, 6), (2, 2), colspan=1, rowspan=1)
    plt.imshow(zOF, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'$Z displacement$', fontsize=ftsize)

    # ------------
    plt.subplot2grid((4, 6), (3, 0), colspan=1, rowspan=1)
    plt.imshow(argOF, cmap='twilight_shifted')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'$arg(OF_x, OF_z)$', fontsize=ftsize)
    plt.tight_layout()

    # --- save fig and close
    plt.savefig(pres, bbox_inches='tight', dpi=1000)
    plt.close()
