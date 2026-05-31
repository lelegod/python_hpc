#!/bin/bash
#BSUB -J task67-head50
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o logs/task67-head50-%J.out
#BSUB -e logs/task67-head50-%J.err

set -euo pipefail

source ~/venvs/hpc/bin/activate

cd ~/hpc/dtu_hpc_project/task6-7
mkdir -p logs report

echo "Python:"
python --version
echo "Starting task 6/7 benchmark on first 50 floorplans..."

/usr/bin/time -v python task6-7.py \
  --n-buildings 50 \
  --workers 8 \
  --worker-counts 1,2,4,8 \
  --engine numba \
  --scheduler dynamic \
  --sample-mode head \
  --chunksize 1 \
  --dynamic-chunksizes 1,2,4,8

echo "Finished."
echo "Outputs should be in task6-7/report/"
