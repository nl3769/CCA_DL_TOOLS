import imageio
import os

# ----------------------------------------------------------------------------------------------------------------------
def get_original(data):
    ''' data contains files containing the string "orignial". We retain only those one. '''
    original=[]
    for name_ in data:
        if 'original' in name_:
            original.append(name_)
    return original
# ----------------------------------------------------------------------------------------------------------------------
def get_scatterers(data):
    ''' data contains files containing the string "scatterers". We retain only those one. '''

    scatterers=[]
    for name_ in data:
        if 'scatterers' in name_ :
            if '.mat' in data:
                print(name_)
            else:
                scatterers.append(name_)

    return scatterers
# ----------------------------------------------------------------------------------------------------------------------
def sort_data(data):
    ''' data contains file name containing value starting from 0 to n. We data in order to 1, 2, ..., n. '''
    dim=len(data)
    sorted_data=[]
    remove_=data[0].split('_')[-1]
    name=data[0].replace('_'+remove_, '')

    for k in range(1, dim+1):
        sorted_data.append(name+'_'+str(k)+'.png')

    return sorted_data
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # folder='/home/laine/cluster/PROJECTS_IO/SIMULATION/SEQUENCE/ANG_DOM_SEQUENCE_FINAL_12_13_2021/phantom' # folder containing images
    # data=os.listdir(folder)
    # # --- image
    # original_name=get_original(data)
    # original_name = sort_data(original_name)
    # # --- scatterers
    # scatterers_name = get_scatterers(data)
    # scatterers_name = sort_data(scatterers_name)
    # # --- load images
    # scatteres=[]
    # original=[]

    # TMP
    start='dicom_ANG_DOM_volume_seq_cp_scatterers_id_seq_'
    end='_id_param_1.png'
    name_=[]
    for k in range(1, 27):
        name_.append(start+str(k)+end)
    tmp_img_=[]
    for k in range(len(name_)):
        tmp_img_.append(imageio.imread(os.path.join('/home/laine/Documents/PROJECTS_IO/SIMULATION/VOLUME/ANG_DOM_VOLUME_SEQ_2022_06_01/phantom/', name_[k])))
    print(1)
    imageio.mimsave('/home/laine/Desktop/motion_3D.gif', tmp_img_, fps=10)

    # for k in range(len(scatterers_name)):
    #     scatt_ = imageio.imread(os.path.join(folder, scatterers_name[k]))
    #     org_ = imageio.imread(os.path.join(folder, original_name[k]))
    #     scatteres.append(scatt_)
    #     original.append(org_)
    #
    # imageio.mimsave('/home/laine/Desktop/scatt.gif', scatteres, fps=10)
    # imageio.mimsave('/home/laine/Desktop/img.gif', original, fps=10)

