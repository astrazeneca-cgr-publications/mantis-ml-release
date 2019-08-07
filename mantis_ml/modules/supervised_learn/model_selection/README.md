## Hyper-parameter fine-tuning
Fine-tuning of hyper-parameters for each classifier with cross-validation. 

The `dnn_grid_search_cv.py` module contains a custom module for fine-tuning neural-net related parameters including the number and size of hidden layers. 
This currently supports simultaneous fine-tuning of up to 2 features and progressively selects optimal parameter values in sequential steps of optimisation.

## Run
```
# Classifier benchmarking
sbatch ./submit_benchmarking.sh ../../../input_configs/classifier_benchmarking.CKD_config.yaml 10

# Grid search for parameters in tree-based classifiers
sbatch ./submit_gridsearch.sh ../../../input_configs/classifier_benchmarking.CKD_config.yaml

# Grid search for parameters in Deep Neural Net
sbatch ./submit_dnn_gridsearch.sh ../../../input_configs/classifier_benchmarking.CKD_config.yaml

```
