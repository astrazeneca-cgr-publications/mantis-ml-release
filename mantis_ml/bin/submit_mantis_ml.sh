#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G

conf=$1
mantis_ml -c $conf -n 10 -i 10

