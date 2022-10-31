#!/bin/sh

nvcc -ptx -arch sm_75 -O3 bf_low_res_image_RF.cu -allow-unsupported-compiler
