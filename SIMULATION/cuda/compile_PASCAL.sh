#!/bin/sh

nvcc -ptx -arch sm_60 -O3 bf_low_res_image.cu
