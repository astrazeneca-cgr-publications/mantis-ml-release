#!/bin/bash

# First: wget [url with Phenolyzer results]; 
# mv out.predicted_gene_scores out.predicted_gene_scores.[phenotype]

phenotype=$1

cat out.predicted_gene_scores.${phenotype} | grep "Normalized score" | cut -f1,4 | sed 's/Normalized score: //' | awk '{print $1","$2}' > ${phenotype}.phenolyzer.ranked_genes.txt
