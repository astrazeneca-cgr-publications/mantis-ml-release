#!/bin/bash
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:0:0

conf=$1
mantis_ml -c $conf -n 30 -i 100

