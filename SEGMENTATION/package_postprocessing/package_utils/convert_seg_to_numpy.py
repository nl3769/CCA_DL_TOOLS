import os

import numpy            as np
from numba              import njit

# ----------------------------------------------------------------------------------------------------------------------------------------------------
@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def convert_seg_to_numpy(LI, MA, borders, pborders):
   
    # --- get min and max idx
    idx_min = []
    idx_max = []

    for key in LI:
        seg_idx = LI[key][0]
        idx_min.append(min(seg_idx))
        idx_max.append(max(seg_idx))
    
    for key in MA:
        seg_idx = MA[key][0]
        idx_min.append(min(seg_idx))
        idx_max.append(max(seg_idx))

    roi_l = max(max(idx_min) + 5, borders[0])
    roi_r = min(min(idx_max) - 5, borders[1])
    
    LI_np = np.zeros(((roi_r-roi_l+1), len(LI)))
    MA_np = np.zeros(((roi_r-roi_l+1), len(LI)))
    idx = list(np.linspace(roi_l, roi_r, (roi_r-roi_l+1)).astype(np.uint8))
    
    for id_seq, key in enumerate(LI.keys()):
        idx_roi = []
        arr = np.array(LI[key][0]).astype(np.uint8)
        arr_val = np.array(LI[key][1])
        
        for id_ in idx:
            idx_roi.append(index(arr, id_)[0])
        
        LI_np[:, id_seq] = arr_val[idx_roi]

    for id_seq, key in enumerate(MA.keys()):
        idx_roi = []
        arr = np.array(MA[key][0]).astype(np.uint8)
        arr_val = np.array(MA[key][1])
        
        for id_ in idx:
            idx_roi.append(index(arr, id_)[0])
        
        MA_np[:, id_seq] = arr_val[idx_roi]

    # --- save borders
    with open(os.path.join(pborders, 'borders.txt'), 'w') as f:
        f.write('left border: ' + str(roi_l) + '\n' + 'right border: ' + str(roi_r))

    return LI_np, MA_np
