import os
import torch
import package_network.utils                as pnu
from package_motion.motionFullImageHandler  import motionFullImgHandler

# ----------------------------------------------------------------------------------------------------------------------
def main():
    
    # --- TODO -> parse parameters
    class param:
        def __init__(self):
            self.MODEL_NAME = 'raft'
            self.PRES = '/home/laine/Documents/PROJECTS_IO/CARODEEPMOTION/NETMOTION/RAFT_pretraining_00/RAFT_PRETRAINED_FLYINGCHAIR_10_PX_FINE_TUNING'
            self.PSAVE = '/home/laine/Desktop/tmp/full_img'
            self.PDATA = '/home/laine/DISK/PROJECTS_IO/MOTION/IN_SILICO/REAL_DATA/prepared_data_IMAGENET'
            self.DROPOUT = 0
            self.CORRELATION_LEVEL = 4
            self.CORRELATION_RADIUS = 4
            self.RESTORE_CHECKPOINT = True
            self.ALTERNATE_COORDINATE = False
            self.PSPLIT = "/home/laine/Documents/PROJECTS_IO/CARODEEPMOTION/SPLITDATA/validation_patients.txt"
            self.PIXEL_WIDTH = 256
            self.PIXEL_HEIGHT = 256
            self.ROI_WIDTH = 5e-3
            self.SHIFT_X = 32
            self.SHIFT_Z = 32

    p = param()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netEncoder, netFlow = pnu.load_model_flow(p)
    netEncoder = netEncoder.to(device)
    netFlow = netFlow.to(device)
    netFlow.eval()
    netEncoder.eval()

    simulation = os.listdir(p.PDATA)
    if 'backup_parameters' in simulation:
        simulation.remove('backup_parameters')
    simulation.sort()
    pdata = p.PDATA
    psave = p.PSAVE
    for simu in simulation:
        p.PDATA = os.path.join(pdata, simu)
        p.PSAVE = os.path.join(psave, simu)
        motion = motionFullImgHandler(p, netEncoder, netFlow, device)
        motion()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
