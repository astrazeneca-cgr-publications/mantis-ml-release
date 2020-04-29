#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:0:0

# mantis-ml
python updated_benchmark_module.py -c ../../conf/ALS_config.yaml -o ../../../out/ALS-production-Full/ -e ../../../misc/overlap-collapsing-analyses/ALS_Science_2015/ALS/Dom_LoF_collapsing_ranking.ALS.sorted.csv -x 41 -y 60

python updated_benchmark_module.py -c ../../conf/Epilepsy_config.yaml -o ../../../out/Epilepsy-production-Full/ -e ../../../misc/overlap-collapsing-analyses/Epilepsy-LancetNeurology_2017/GGE/primary_collapsing_ranking.GGE.csv -x 42 -y 60

python updated_benchmark_module.py -c ../../conf/CKD_config.yaml -o ../../../out/CKD-production-Full/ -e ../../../misc/overlap-collapsing-analyses/CKD_JASN_2019/CKD/v-AURORA-CUMC-all_dom_ultrarare_OO_collapsing_ranking.CKD.csv -x 67 -y 60


# Phenolyzer  -- (similarly for ToppGene and ToppNet, just replacing the Phenolyzer arg)
python updated_benchmark_module.py -c ../../conf/CKD_config.yaml -o ../../../out/CKD-production-Full/ -e ../../../misc/overlap-collapsing-analyses/CKD_JASN_2019/CKD/v-AURORA-CUMC-all_dom_ultrarare_OO_collapsing_ranking.CKD.csv -x 67 -y 60 -b Phenolyzer -p CKD

python updated_benchmark_module.py -c ../../conf/ALS_config.yaml -o ../../../out/ALS-production-Full/ -e ../../../misc/overlap-collapsing-analyses/ALS_Science_2015/ALS/Dom_LoF_collapsing_ranking.ALS.sorted.csv -x 41 -y 60 -b Phenolyzer -p ALS

python updated_benchmark_module.py -c ../../conf/Epilepsy_config.yaml -o ../../../out/Epilepsy-production-Full/ -e ../../../misc/overlap-collapsing-analyses/Epilepsy-LancetNeurology_2017/GGE/primary_collapsing_ranking.GGE.csv -x 42 -y 60 -b Phenolyzer -p Epilepsy



## Synonymous models
ALS: Dom_not_benign_collapsing_ranking.ALS.sorted.csv
