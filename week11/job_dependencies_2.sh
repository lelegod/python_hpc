#!/bin/bash
#BSUB -J dependent_job
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -W 00:10
#BSUB -R "rusage[mem=1GB]"
#BSUB -w "ended(21241475)"
#BSUB -o dependent_%J.out
#BSUB -e dependent_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

echo "Running after ALL elements of array 21241475 have exited (any reason)"
