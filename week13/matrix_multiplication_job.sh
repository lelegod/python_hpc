#!/bin/bash
#BSUB -J matmuls
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 00:10
#BSUB -o matmuls_%J.out
#BSUB -e matmuls_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Enable NumPy multi-threading — set to match the number of allocated cores
NUM_THREADS=8
OMP_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
OPENBLAS_NUM_THREADS=$NUM_THREADS

python matmul.py
