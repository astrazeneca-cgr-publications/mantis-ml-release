## Run
```
# Classifier benchmarking
sbatch ./submit_benchmarking.sh ../../../input_configs/classifier_benchmarking.CKD_config.yaml 10

# Grid search for parameters in tree-based classifiers
sbatch ./submit_gridsearch.sh ../../../input_configs/classifier_benchmarking.CKD_config.yaml

# Grid search for parameters in Deep Neural Net
sbatch ./submit_dnn_gridsearch.sh ../../../input_configs/classifier_benchmarking.CKD_config.yaml

```
