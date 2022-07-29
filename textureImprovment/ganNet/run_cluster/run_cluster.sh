#!/bin/bash

parameters=("lbd-GAN:1-over-10_lbd-pxl:1_kernel:3-3_32_loss:L1.py"
            "lbd-GAN:1-over-10_lbd-pxl:1_kernel:3-3_32_loss:L1L2.py"
            "lbd-GAN:1-over-10_lbd-pxl:1_kernel:3-3_32_loss:L2.py"
            "lbd-GAN:1-over-30_lbd-pxl:1_kernel:3-3_32_loss:L1.py"
            "lbd-GAN:1-over-30_lbd-pxl:1_kernel:3-3_32_loss:L1L2.py"
            "lbd-GAN:1-over-30_lbd-pxl:1_kernel:3-3_32_loss:L2.py"
            "lbd-GAN:1-over-30_lbd-pxl:1_kernel:7-7_32_loss:L1L2.py"
            "lbd-GAN:1-over-30_lbd-pxl:1_kernel:3-3_64_loss:L1L2.py"
            )

exp=("1-over-10_pxl:1_kernel:3-3_loss:L1"
     "1-over-10_pxl:1_kernel:3-3_loss:L1L2"
     "1-over-10_pxl:1_kernel:3-3_loss:L2"
     "1-over-30_pxl:1_kernel:3-3_loss:L1"
     "1-over-30_pxl:1_kernel:3-3_loss:L1L2"
     "1-over-30_pxl:1_kernel:3-3_loss:L2"
     "1-over-30_pxl:1_kernel:7-7_loss:L1L2"
     "1-over-30_pxl:1_kernel:3-3_loss:L1L2"
     )


for idx in ${!parameters[@]}; do
  qsub -N ${exp[$idx]} -v log_name=${exp[$idx]},param=${parameters[$idx]} run_cluster.pbs
  echo ${parameters[$idx]}
  echo ${exp[$idx]}
done
