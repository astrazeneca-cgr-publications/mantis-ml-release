#!/bin/bash

methods=(Rec_coding
	 Rec_LoF
	 Rec_not_benign
	 Dom_coding 
	 Dom_LoF
	 Dom_not_benign)

collapsing_file="aaa3650-Cirulli-Table-S6.csv"


for method in ${methods[@]}; do
	echo $method

	out_file=ALS/${method}_collapsing_ranking.ALS.csv
	cat $collapsing_file | grep $method | cut -d',' -f1,2 > $out_file  #sort -t ',' -k2 -n
done
