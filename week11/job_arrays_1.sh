#!/bin/bash
#BSUB -J array[1-10]
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -W 00:10
#BSUB -R "rusage[mem=1GB]"
#BSUB -o array_%J_%I.out
#BSUB -e array_%J_%I.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

echo "Running array element $LSB_JOBINDEX"
