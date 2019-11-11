#!/bin/bash

# First: wget [url with ToppNet results]; 

phenotype=$1

tail -n+2 ToppNet_${phenotype}.csv | cut -d',' -f3,5 > ${phenotype}.toppnet.ranked_genes.txt
