import os
import numpy        as np
def save_seg(LI, MA, psave, name):
    
   
    # --- save LI
    idx = np.nonzero(LI)
    with open(os.path.join(psave, name + "-LI.txt"), 'w') as f:
        for i in np.nditer(idx):
            f.write(str(i) + ' ' + str(LI[i]) + '\n')
    
    # --- save MA
    idx = np.nonzero(MA)
    with open(os.path.join(psave, name + "-MA.txt"), 'w') as f:
        for i in np.nditer(idx):
            f.write(str(i) + ' ' + str(MA[i]) + '\n')
