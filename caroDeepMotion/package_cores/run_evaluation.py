import package_metrics.metricsHandler as mh

# ----------------------------------------------------------------------------------------------------------------------
def main_dl_full_image():
    """ Compute error for deep learning approach on full images. """

    pdata = '/run/media/laine/DISK/PROJECTS_IO/MOTION/PREDICTION/DL_METHOD_FULL_IMAGES'
    pres = '/run/media/laine/DISK/PROJECTS_IO/MOTION/EVALUATION/DL_FULL_IMAGES'
    nmethod = 'DL full img'
    metricsHandler = mh.evaluationDLFullImg(pdata, pres, nmethod)
    metricsHandler.compute_angle_error(metricsHandler.patients.keys())
    metricsHandler.compute_EPE(metricsHandler.patients.keys())
    metricsHandler.plot_metrics()

# ----------------------------------------------------------------------------------------------------------------------
def main_dg():
    """ Compute error for damien garcia's method. """

    pdata = '/run/media/laine/DISK/PROJECTS_IO/MOTION/PREDICTION/DG_METHOD'
    pres = '/run/media/laine/DISK/PROJECTS_IO/MOTION/EVALUATION/DG'
    metricsHandler = mh.evaluationDG(pdata, pres, 'DG')
    metricsHandler()

# ----------------------------------------------------------------------------------------------------------------------
def main_dl_individual_patches():
    """ Compute error for deep learning approach on patches. """

    pdata = '/run/media/laine/DISK/PROJECTS_IO/MOTION/IN_SILICO/REAL_DATA/database_training_IMAGENET'
    pres = '/run/media/laine/DISK/PROJECTS_IO/MOTION/EVALUATION/DL_METHOD_PATCHES'
    psplit = "/run/media/laine/DISK/PROJECTS_IO/MOTION/IN_SILICO/REAL_DATA/SPLIDATA/validation_patients.txt"

    class param:
        def __init__(self):
            self.MODEL_NAME = 'raft'
            self.PRES = '/run/media/laine/DISK/PROJECTS_IO/MOTION/NETWORK_TRAINING/RAFT_PRETRAINED_FLYINGCHAIR_10_PX_tmp'
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

    main_dl_full_image()
    main_dl_individual_patches()
    main_dg()