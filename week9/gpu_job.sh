#!/bin/bash
#BSUB -q c02613
#BSUB -J add_kernel
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:10
#BSUB -o add_kernel_%J.out
#BSUB -e add_kernel_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python add_kernel.py
