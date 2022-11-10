#!/bin/sh

nvcc -ptx -arch sm_75 -O3 *.cu -allow-unsupported-compiler

mkdir bin
mv *.ptx ./bin
