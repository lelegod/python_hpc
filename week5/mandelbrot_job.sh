#!/bin/bash
#BSUB -J mandelbrot
#BSUB -q hpc
#BSUB -W 30
#BSUB -R "rusage[mem=2000MB]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o mandelbrot_%J.out
#BSUB -e mandelbrot_%J.err

export NUMBA_DISABLE_CUDA=1
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

echo "num_proc,time" > timing_results.csv
for n in 1 2 4 8 16; do
    python mandelbrot2.py $n >> timing_results.csv
done

python plot_speedup.py
