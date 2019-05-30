#!/bin/sh
#SBATCH -J boruta 
#SBATCH -o boruta.out 
#SBATCH -e boruta.err 
#SBATCH --time=24:00:00 
#SBATCH --mem-per-cpu=1G  
#SBATCH --cpus-per-task=30  

python run_boruta.py
