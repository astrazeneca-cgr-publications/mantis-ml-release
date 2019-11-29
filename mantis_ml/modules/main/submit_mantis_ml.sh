#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:0:0

conf="../../conf/CKD_config.yaml" #$1

mantisml -c $conf -o ../../../out/CKD-example_bal-ratio_2 -n 30 -i 10
