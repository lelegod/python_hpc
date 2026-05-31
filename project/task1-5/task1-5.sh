#!/bin/bash
#BSUB -J task15
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=16GB]"
#BSUB -W 01:00
#BSUB -o logs/task15-%J.out
#BSUB -e logs/task15-%J.err

set -eo pipefail

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd ~/Documents/dtu_hpc_project/task1-5
mkdir -p logs report

export DTU_HPC_LOAD_DIR="/dtu/projects/02613_2025/data/modified_swiss_dwellings/"

echo "Python:"
python --version
echo "Starting tasks 1-5..."

/usr/bin/time -v python task1-5.py

echo "Finished."
echo "Outputs in report/"