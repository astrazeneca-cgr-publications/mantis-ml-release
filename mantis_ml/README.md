## mantis-ml

A high-level machine learning framework for gene prioritisation implemented on top of `scikit-learn` and `keras`.

<br>

- Usage: 
###
```
cd bin
./run_mantis_ml.sh [-c CONFIG_FILE]
```

### SLURM submission script
```
cd bin
sbatch ./submit_mantis_ml.sh [-h] [-u|--user Unix_username] [-c|--config CONFIG_FILE] [-m|--mem MEMORY]
      			 [-t|--time TIME] [-n|nthredas NUM_THREADS]
```

#### Examples
```
# Run locally  
./run_mantis_ml.sh -c ../conf/config.yaml

# generic (SLURM)
sbatch -o generic.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh -u [my_unix_username] -c input_configs/Generic_config.yaml -m 12G -n 10

# CKD (SLURM)
sbatch -o ckd.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh -u [my_unix_username] -c input_configs/CKD_config.yaml

# Epilepsy (SLURM)
sbatch -o epilepsy.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh -u [my_unix_username] -c input_configs/Epilepsy_config.yaml

# ALS (SLURM)
sbatch -o als.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh -u [my_unix_username] -c input_configs/ALS_config.yaml
```
