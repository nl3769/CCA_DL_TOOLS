from metricsHandler import evaluation

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    pdata = '/home/laine/Desktop/test_DZ_01'
    pres = '/home/laine/Documents/PROJECTS_IO/CARODEEPMOTION/MOTION_ANALYSIS/evaluation'

    metricsHandler = evaluation(pdata, pres)
    metricsHandler.compute_EPE()
    metricsHandler.compute_angle_error()
    metricsHandler.plot_metrics()
