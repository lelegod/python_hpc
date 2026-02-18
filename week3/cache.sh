#!/bin/bash
#BSUB -J python
#BSUB -q hpc
#BSUB -W 10
#BSUB -R "rusage[mem=16384MB]"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -o python%J.out
#BSUB -e python_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python cache.py