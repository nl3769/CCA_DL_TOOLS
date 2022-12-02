#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepSeg
cd $WD
PYTHONPATH=$WD python package_example/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f0.py
PYTHONPATH=$WD python package_example/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f1.py
PYTHONPATH=$WD python package_example/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f2.py
PYTHONPATH=$WD python package_example/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f3.py
PYTHONPATH=$WD python package_example/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f4.py
PYTHONPATH=$WD python package_example/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f5.py
PYTHONPATH=$WD python package_example/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f6.py
PYTHONPATH=$WD python package_example/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f7.py
PYTHONPATH=$WD python package_example/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f8.py
PYTHONPATH=$WD python package_example/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f9.py
