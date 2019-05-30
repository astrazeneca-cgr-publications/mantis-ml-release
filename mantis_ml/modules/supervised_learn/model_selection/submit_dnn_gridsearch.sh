#!/bin/bash 
#SBATCH -J dnn_gridsearch 
#SBATCH -o out_dnn_gridsearch.txt 
#SBATCH -e err_dnn_gridsearch.txt 
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G

config_file=$1

python -u dnn_grid_search_cv.py $config_file
