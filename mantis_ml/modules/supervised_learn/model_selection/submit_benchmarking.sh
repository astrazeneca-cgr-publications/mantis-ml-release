#!/bin/bash 
#SBATCH -J benchmark 
#SBATCH -o out_benchmark.txt 
#SBATCH -e err_benchmark.txt 
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G

config_file=$1
iterations=$2

python benchmark_all_classifiers.py $config_file $2
