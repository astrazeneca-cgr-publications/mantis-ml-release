#!/bin/bash

config_file=$1
top_ratio=$2
remove_seed_genes=$3 #0
show_full_xaxis=$4 #0


python overlap_collapsing_w_mantis_predictions.py $config_file $top_ratio $remove_seed_genes $show_full_xaxis

python get_consensus_novel_predictions.py $config_file Novel $top_ratio
python get_consensus_novel_predictions.py $config_file Known $top_ratio
