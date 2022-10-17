import os

import numpy                    as np
import matplotlib.pyplot        as plt

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def compute_IMC(LI, MA, pres = None, display=True):
    IMC = MA - LI
    mean_IMC = np.mean(IMC, axis=0)

    display = True
    if display and pres is not None:
        plt.imshow(IMC*1e6)
        plt.colorbar()
        plt.set_cmap("hot")
        plt.title('IMC in um')
        plt.savefig(os.path.join(pres, 'IMC_hot.png'))
        plt.close()
        
        plt.plot(mean_IMC*1e6)
        plt.title('IMC in um')
        plt.savefig(os.path.join(pres, 'IMC_mean.png'))
        plt.close()
