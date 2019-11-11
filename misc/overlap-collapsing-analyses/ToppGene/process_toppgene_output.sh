#!/bin/bash

# First: wget [url with ToppGene results]; 

phenotype=$1

tail -n+2 ToppGeneData_${phenotype}.csv | cut -d',' -f2,29 | sed 's/\"//g' > ${phenotype}.toppgene.ranked_genes.txt
