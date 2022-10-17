from PIL            import Image
from icecream       import ic
import numpy        as np
import nibabel      as nib
import pickle       as pkl

# ----------------------------------------------------------------------------------------------------------------------
def write_image(I, pres):

    I = I.astype(np.uint8)
    im = Image.fromarray(I)
    im.save(pres, format="png")

# ----------------------------------------------------------------------------------------------------------------------
def mk_nifty(seq, pres, CF):

    seq_array = np.zeros((len(seq),) + seq[0].shape)
    for i in range(len(seq)):
        seq_array[i,] = seq[i]

    affine = np.eye(4)
    affine[0, 0] = CF
    affine[1, 1] = CF
    ni_seq = nib.Nifti1Image(seq_array, affine)
    print(ni_seq.header)
    nib.save(ni_seq, pres)

# ----------------------------------------------------------------------------------------------------------------------
def save_np_to_pickle(arr, path):
    with open(path, 'wb') as f:
        pkl.dump(arr, f)
