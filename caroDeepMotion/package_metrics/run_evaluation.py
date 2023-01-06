import metricsHandler as mh

# ----------------------------------------------------------------------------------------------------------------------
def main_dl_full_image():
    """ Compute error for deep learning approach on full images. """

    pdata = '/home/laine/Desktop/tmp/full_img'
    pres = '/home/laine/Desktop/test_motion_eval/dl_full_img'
    nmethod = 'DL full img'
    metricsHandler = mh.evaluationDLFullImg(pdata, pres, nmethod)
    metricsHandler.compute_angle_error(metricsHandler.patients.keys())
    metricsHandler.compute_EPE(metricsHandler.patients.keys())
    metricsHandler.plot_metrics()

# ----------------------------------------------------------------------------------------------------------------------
def main_dg():
    """ Compute error for damien garcia's method. """

    pdata = '/home/laine/Documents/PROJECTS_IO/MOTION/BASELINE/DG_METHOD'
    pres = '/home/laine/Desktop/test_motion_eval/dg_full_img'
    metricsHandler = mh.evaluationDG(pdata, pres, 'DG')
    metricsHandler()

# ----------------------------------------------------------------------------------------------------------------------
def main_dl_individual_patches():
    """ Compute error for deep learning approach on patches. """

    pdata = '/home/laine/DISK/PROJECTS_IO/MOTION/IN_SILICO/REAL_DATA/database_training_IMAGENET'
    pres = '/home/laine/Desktop/test_motion_eval/dl_patches'
    psplit = "/home/laine/Documents/PROJECTS_IO/CARODEEPMOTION/SPLITDATA/validation_patients.txt"

    class param:
        def __init__(self):
            self.MODEL_NAME = 'raft'
            self.PRES = '/home/laine/Documents/PROJECTS_IO/CARODEEPMOTION/NETMOTION/RAFT_pretraining_00/RAFT_PRETRAINED_FLYINGCHAIR_10_PX_tmp'
            self.DROPOUT = 0
            self.CORRELATION_LEVEL = 4
            self.CORRELATION_RADIUS = 4
            self.RESTORE_CHECKPOINT = True
            self.ALTERNATE_COORDINATE = False
            self.METHOD = "DL patches"

    p = param()
    metricsHandler = mh.evaluationDL(pres, pdata, psplit, p)
    metricsHandler()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """ Run evaluation for motion estimation. It computes End point Error and Angle Error. """

    main_dg()
    main_dl_individual_patches()
    main_dl_full_image()