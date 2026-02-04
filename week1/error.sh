#!/bin/bash
#BSUB -J sleeper
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=512GB]"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o sleeper_%J.out
#BSUB -e sleeper_%J.err

lscpu
sleep 60
