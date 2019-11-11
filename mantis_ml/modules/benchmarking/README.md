# ====== Benchmarking for mantis-ml ======
## CKD
```
 ./run_collapsing_mantis_overlap.sh ../../conf/CKD_config.yaml 0
```

## Epilepsy
```
 ./run_collapsing_mantis_overlap.sh ../../conf/Epilepsy_config.yaml 0
```

## ALS
```
 ./run_collapsing_mantis_overlap.sh ../../conf/ALS_config.yaml 0
```


# ====== Benchmarking for external tools ======
# CKD
./run_collapsing_mantis_overlap.sh ../../conf/CKD_config.yaml 1 Phenolyzer
./run_collapsing_mantis_overlap.sh ../../conf/CKD_config.yaml 1 ToppGene
./run_collapsing_mantis_overlap.sh ../../conf/CKD_config.yaml 1 ToppNet

# Epilepsy
./run_collapsing_mantis_overlap.sh ../../conf/Epilepsy_config.yaml 1 Phenolyzer
./run_collapsing_mantis_overlap.sh ../../conf/Epilepsy_config.yaml 1 ToppGene
./run_collapsing_mantis_overlap.sh ../../conf/Epilepsy_config.yaml 1 ToppNet

# ALS
./run_collapsing_mantis_overlap.sh ../../conf/ALS_config.yaml 1 Phenolyzer
./run_collapsing_mantis_overlap.sh ../../conf/ALS_config.yaml 1 ToppGene
./run_collapsing_mantis_overlap.sh ../../conf/ALS_config.yaml 1 ToppNet
