#!/bin/bash 
#SBATCH -J extra_trees
#SBATCH -o extra_trees_out.txt 
#SBATCH -e extra_trees_err.txt 
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=100000  
#SBATCH --gres=gpu:volta:4

python pu_learning.py

