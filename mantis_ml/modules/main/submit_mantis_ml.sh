#!/bin/bash
#SBATCH -o ckd-fast.out
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:0:0


# Various Test runs
iterations=10

python __main__.py -c ../../conf/CKD_config.yaml -o CKD-fast -n 10 -i $iterations -f 


#python __main__.py -c ../../conf/CKD_config.yaml -o CKD-et_rf -n 10 -i $iterations -m et,rf


#python __main__.py -c ../../conf/CKD_config.yaml -o CKD-stacking -n 10 -i $iterations -m stack

