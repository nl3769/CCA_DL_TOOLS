import os
import glob
import imageio as iio
import cv2

# ----------------------------------------------------------------------------------------------------------------------
def main():

    pdata = "/home/laine/Desktop/ICCVG_SEQ/tech_268/SEQ"
    fnames = sorted(glob.glob(os.path.join(pdata, "*_id_*")))

    I = []
    for fname in fnames[:100]:
        p_scatt = os.path.join(pdata, fname, 'phantom', '2D_phantom.png')
        I.append(iio.imread(p_scatt))

    iio.mimsave(os.path.join(pdata, 'scatt_2D.gif'), I, fps=40)

    # pdata = "/home/laine/Desktop/ICCVG_SEQ/tech_268"
    # fnames = sorted(glob.glob(os.path.join(pdata, "*_id_*")))
    #
    I = []

    for fname in fnames[:100]:
        p_scatt = os.path.join(pdata, fname, 'bmode_result','RF')
        print(p_scatt)
        p_scatt = glob.glob(os.path.join(p_scatt, "*bmode.png"))[0]
        I.append(iio.imread(p_scatt))


    iio.mimsave(os.path.join(pdata, 'in-silico.gif'), I, fps=40)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()