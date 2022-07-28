import os
import argparse
import importlib
import shutil
import imageio                                  as io
import package_utils.get_fname                  as pugf
import package_utils.load_data                  as puld
import package_utils.load_model                 as pulm
import package_processing.apply_gan             as ppag
import package_utils.writter                    as puw
from package_processing.histogram_extension     import histogram_extension
from package_utils.save_seg                     import save_seg
from package_processing.adapt_seg               import adapt_segmentation
from package_processing.add_seg                 import add_seg

# -----------------------------------------------------------------------------------------------------------------------
def main():

    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    fnames = os.listdir(p.PDATA)
    fnames.sort()
    # --- list to store data
    inames      = []
    LInames     = []
    MAnames     = []
    imgnames    = []
    for patient in fnames:
        inames.append(pugf.get_fname(dir=os.path.join(p.PDATA, patient), sub_str='_bmode.png', fold=os.path.join('bmode_result', 'RF')))
        LInames.append(pugf.get_fname(dir=os.path.join(p.PDATA, patient), sub_str='LI.mat', fold='phantom'))
        MAnames.append(pugf.get_fname(dir=os.path.join(p.PDATA, patient), sub_str='MA.mat', fold='phantom'))
        imgnames.append(pugf.get_fname(dir=os.path.join(p.PDATA, patient), sub_str='image_information', fold='phantom')) # -> to get calibration factor

    # --- load model
    model = pulm.load_model(p)

    # --- load images and copy them in RES folder -> it can be stored in RAM
    I_org = []
    I_gan = []
    LI = []
    MA = []
    for id, iname in enumerate(inames):
        
        # --- get original image
        I = puld.load_image(iname)

        # --- apply GAN
        Ig = ppag.apply_gan(model, p.DIM_IMG_GAN, I, p.INTERVAL)
        
        # --- store current image
        I = histogram_extension(I, (0, 255))
        Ig = histogram_extension(Ig, (0, 255))
        I_org.append(I)
        I_gan.append(Ig)
        
        # --- get seg
        LI_ = puld.load_seg(LInames[id], 'LI_val')
        MA_ = puld.load_seg(MAnames[id], 'MA_val')
        LI_, MA_ = adapt_segmentation(LI_, MA_, I.shape)
        LI.append(LI_)
        MA.append(MA_)
        
        # --- get CF
        if id == 0:
            CF = puld.load_CF(imgnames[id])  # only one time is enought because the calibration factor is the same for the full sequence 

        # --- save results (without segmentation)
        name_= iname.split('/')[-1]
        puw.write_image(I,  os.path.join(p.PORG, name_))
        puw.write_image(Ig, os.path.join(p.PGAN_OUTPUT, name_))

        # --- save seg
        save_seg(LI_, MA_, p.PSEG_GT, name_.split('.')[0])
    
    # --- save the sequence (nii format)
    puw.mk_nifty(I_org, os.path.join(p.PSEQ, 'seq_org.nii'), CF)
    puw.mk_nifty(I_gan, os.path.join(p.PSEQ, 'seq_gan.nii'), CF)

    # --- mk gif (without segmentation)
    io.mimsave(os.path.join(p.PGIF, 'GAN.gif'), I_gan, fps = 10)
    io.mimsave(os.path.join(p.PGIF, 'ORG.gif'), I_org, fps = 10)

    # --- add segmentation
    add_seg(LI, MA, I_org, I_gan)
    
    # --- mk gif (without segmentation)
    io.mimsave(os.path.join(p.PGIF, 'seg_GAN.gif'), I_gan, fps = 10)
    io.mimsave(os.path.join(p.PGIF, 'seg_ORG.gif'), I_org, fps = 10)
    

# -----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
