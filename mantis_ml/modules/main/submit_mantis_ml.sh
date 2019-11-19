#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:0:0

conf=$1
mantisml -c $conf -o ./Test-run -n 30 -i 2
