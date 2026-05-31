#!/bin/bash
#BSUB -B -N
#BSUB -J python
#BSUB -W 20
#BSUB -n 4
#BSUB -R "rusage[mem=256MB]"
#BSUB -o python_%J.out
#BSUB -e python_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q c02613
#BSUB -R "span[hosts=1]"
# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
# Run Python script
python cupy_proj_new.py 50
#nsys profile -o cupy_profile_50 --stats=true python cupy_proj.py 50