'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import matplotlib.pyplot        as plt
import numpy                    as np

# ----------------------------------------------------------------------------------------------------------------------
def make_figure(I1, I2, M1, M2, I2_warpped, xOF, zOF, argOF, normOF, pres):

    ftsize = 3.5

    rec = [32, 32]
    x_mean, y_mean = int(I1.shape[1]/2), int(I2.shape[0]/2)

    I1 = np.array(I1).astype(np.int16)
    I2 = np.array(I2).astype(np.int16)
    M1 = np.array(M1).astype(np.int16)
    M2 = np.array(M2).astype(np.int16)

    I2_disp = I2.copy()
    I1_disp = I1.copy()
    I2_warpped_disp = I2_warpped.copy()

    for id_x in range(x_mean - rec[0], x_mean + rec[0]):
        I2_disp[y_mean - rec[1], id_x] = 255
        I2_disp[y_mean + rec[1], id_x] = 255
        I1_disp[y_mean - rec[1], id_x] = 255
        I1_disp[y_mean + rec[1], id_x] = 255
        I2_warpped_disp[y_mean - rec[1], id_x] = 255
        I2_warpped_disp[y_mean + rec[1], id_x] = 255

    for id_y in range(y_mean - rec[1], y_mean + rec[1]):
        I2_disp[id_y, x_mean - rec[1]] = 255
        I2_disp[id_y, x_mean + rec[1]] = 255
        I1_disp[id_y, x_mean - rec[1]] = 255
        I1_disp[id_y, x_mean + rec[1]] = 255
        I2_warpped_disp[id_y, x_mean - rec[1]] = 255
        I2_warpped_disp[id_y, x_mean + rec[1]] = 255

    plt.rcParams['text.usetex'] = True
    plt.figure()

    # ------ IMAGE
    plt.subplot2grid((4, 6), (0, 0), colspan=1, rowspan=1)
    plt.imshow(I1_disp, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'I1', fontsize=ftsize)
    # ------------
    plt.subplot2grid((4, 6), (0, 1), colspan=1, rowspan=1)
    plt.imshow(I2_disp, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'I2', fontsize=ftsize)
    # ------------
    plt.subplot2grid((4, 6), (0, 2), colspan=1, rowspan=1)
    diff = I2 - I1
    plt.imshow(diff[y_mean-rec[1]:y_mean+rec[1], x_mean-rec[0]:x_mean+rec[0]], cmap='jet')
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

    # ------ I2 WARPPED
    plt.subplot2grid((4, 6), (3, 0), colspan=1, rowspan=1)
    plt.imshow(I2_warpped_disp, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'$I2_{warp.}$', fontsize=ftsize)
    # ------------
    plt.subplot2grid((4, 6), (3, 1), colspan=1, rowspan=1)
    plt.imshow(I2_disp, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'$I2$', fontsize=ftsize)
    # ------------
    plt.subplot2grid((4, 6), (3, 2), colspan=1, rowspan=1)
    diff = I2_warpped-I2
    plt.imshow(diff[y_mean-rec[1]:y_mean+rec[1], x_mean-rec[0]:x_mean+rec[0]], cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.title(r'$I2_{warp.} - I2$', fontsize=ftsize)

    # --- save fig and close
    plt.savefig(pres, bbox_inches='tight', dpi=1000)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------