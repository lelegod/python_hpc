#!/bin/bash
#BSUB -J task11-array[1-4]
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=8GB]"
#BSUB -W 01:30
#BSUB -o logs/task11-array-%J-%I.out
#BSUB -e logs/task11-array-%J-%I.err

set -euo pipefail

source ~/venvs/hpc/bin/activate

cd ~/hpc/dtu_hpc_project/task11
mkdir -p logs report

echo "Python:"
python --version
echo "Starting task 11 array worker ${LSB_JOBINDEX}..."

/usr/bin/time -v python task11-array-worker.py \
  --workers 8 \
  --chunksize 1

echo "Finished task 11 array worker ${LSB_JOBINDEX}."
