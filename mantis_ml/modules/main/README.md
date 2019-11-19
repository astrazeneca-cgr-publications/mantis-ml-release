# Benchmark datasets
```
sbatch -o CKD.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/CKD_config.yaml;

sbatch -o ALS.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/ALS_config.yaml;

sbatch -o Epilepsy.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/Epilepsy_config.yaml
```

-----

# Additional runs for mantis-ml gene prioritisation atlas
```
sbatch -o immunodeficiency.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/Immunodeficiency_config.yaml

sbatch -o cardiovascular.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/Cardiovascular_config.yaml

sbatch -o autism.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/Autism_config.yaml

sbatch -o alzheimers.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/Alzheimers_config.yaml

sbatch -o pulmonary_diseases.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/PulmonaryDiseases_config.yaml

sbatch -o respiratory_diseases.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/RespiratoryDiseases_config.yaml

sbatch -o intellectual_disability.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/IntellectualDisability_config.yaml

sbatch -m 12G -o generic.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh ../../conf/Generic_config.yaml
```
