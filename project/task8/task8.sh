#!/bin/bash
#BSUB -J task8-cuda
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1] rusage[mem=8GB]"
#BSUB -W 00:30
#BSUB -o logs/task8-%J.out
#BSUB -e logs/task8-%J.err

set -euo pipefail

module load cuda/12.6

CUDA_BASE=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$CUDA_BASE/nvvm/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDA_BASE
export CUDA_LAUNCH_BLOCKING=1

source ~/project/venv/bin/activate

echo "Python:  $(python --version)"
echo "Numba:   $(python -c 'import numba; print(numba.__version__)')"
echo "GPU:     $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NOT SET}"

cd ~/project/task8
mkdir -p logs report

echo ""
echo "Starting Task 8 – Numba CUDA Jacobi on 10 buildings..."
/usr/bin/time -v python task8.py \
  --n-buildings 10 \
  --max-iter 20000 \
  --sample-mode head

echo "Done. Results in task8/report/"
