#!/bin/bash
#BSUB -J python
#BSUB -q hpc
#BSUB -W 15
#BSUB -R "rusage[mem=2000MB]"
#BSUB -n 10
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o logs/pi/python_%J.out
#BSUB -e logs/pi/python_%J.err

mkdir -p logs/pi

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

time python pi_serial.py
time python pi_parallel.py
time python pi_chunked.py