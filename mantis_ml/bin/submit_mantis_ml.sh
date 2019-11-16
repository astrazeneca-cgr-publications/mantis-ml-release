#!/bin/bash
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:0:0

conf=$1
mantis_ml -c $conf -n 20 -i 10

