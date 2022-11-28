#!/bin/sh

nvcc -ptx -arch sm_60 -O3 *.cu -allow-unsupported-compiler
mkdir bin
mv *.ptx ./bin
