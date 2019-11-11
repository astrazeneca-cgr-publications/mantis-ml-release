#!/bin/bash

config_file=$1
run_external_benchmark=$2
benchmark_tool=$3

top_ratio=0.05


python benchmarking_collapsing_hypergeom_enrichment.py $config_file $run_external_benchmark $benchmark_tool


# Get consensus novel and known genes from mantis-ml predictions
if [ $run_external_benchmark == "0" ]; then
	python get_consensus_novel_predictions.py $config_file Novel $top_ratio 
	python get_consensus_novel_predictions.py $config_file Known $top_ratio 
fi
