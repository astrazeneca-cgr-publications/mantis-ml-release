## Stochastic semi-supervised learning
The `pu_learning` module implements the stochastic semi-supervised framework used for extracting gene rankings as probability prediction scores.

## Example runs
- CKD, Epilepsy arguments can be any string descriptive of a job -- they are not restricted by any dictionary and do not affect the job running, this is only defined by the config.yaml used'
```
./submit_all_clf.sh CKD

./submit_all_clf.sh Epilepsy
```

## Tweak parameters
- max_workers = 30 # Deafult for CPU/GPU submission: 30 or 40  ; for Generic disease classifier: 10     
- iterations = 20 #default: 100; use 20 or 30 for ALS initially

