#!/bin/sh

nvcc -ptx -arch sm_75 -O3 bf_low_res_image.cu
