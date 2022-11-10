#!/bin/sh

nvcc -ptx -arch sm_60 -O3 *.cu
mkdir bin
mv *.ptx ./bin
