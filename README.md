# mantis-ml (v 1.0)

- [Introduction](#introduction) 
- [Installation](#installation) 
- [Run](#run) 
- [Output](#output) 
- [Workflow overview](#workflow-overview) 
- [Package contents](#package-contents) 




Introduction
============
`mantis-ml` is a disease-agnostic gene prioritisation framework, built on top of `scikit-learn` and `keras`/`tensorflow`.  
`mantis-ml` takes its name from the Greek word 'μάντης' which means 'fortune teller', 'predicter'.

<br>

Installation
============
### Requirements:
- Python (Python3, tested with v3.6.7)
- R (tested with v3.5.1)

<br>

**1. Download `mantis-ml-release` GitHub repository:**
```
git clone https://github.com/astrazeneca-cgr-publications/mantis-ml-release.git
```

**(Optional - for Anaconda users):**
- [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
- Create a new environment with clean Python and R installations:
```
conda create -n mantis_ml python=3.6 r
```

<br>

**2. Install `Python` library dependencies:**
```
pip install -r requirements.txt
```

or with `conda`:
```
conda install --file requirements.txt
```

<br>

**3. Install `mantis-ml` package:**
```
cd mantis-ml-release
python setup.py install
```

and add based `mantis-ml-release` dir to `PYTHONPATH`:
```
# in ~/.basrhc add:
export PYTHONPATH=[FULL_PATH_TO_DIR]/mantis-ml-release:$PYTHONPATH
```


<br>

**4. Install `R` library dependencies:**
```
# R 
> install.packages('Boruta')
```

<br><br>


Run
===

### 1. Prepare config file
-------------------

#### Basic parameters:
|__run__ parameters| Description|
| --- | --- |
|`Tissue`| primary tissue affected by disease|
|`additional_tissues`| other tissues affected by disease|
|`seed_include_terms`| patterns matching HPO phenotypes for annotation of known disease genes (seed genes)|
|`additional_include_terms`| patterns used alongside `seed_include_terms` for disease-specific feature extraction|
|`exclude_terms`| string patterns to exclude during seed gene selection and/or disease-specific feature extraction|
|`phenotype`| user-defined descriptive term for the disease/phenotype|
|`run_id`| output folder-name suffix|

<br/>

|__pu__ parameters| Description|
| --- | --- |
|`classifiers`| define list of classifiers to use for Positive-Unlabelled learning|
|`iterations`| number _L_ of stochastic iterations|
|`nthreads`| number of threads to use (optimally assign one CPU per thread)|

<br/>

|__run_steps__ parameters| Description|
| --- | --- |
|`run_boruta`| _True_/_False_ to run/or not the Boruta feature importance estimation algorithm|
|`run_unsupervised`| _True_/_False_ to run/or not unsupervised learning methods (PCA, t-SNE and UMAP) during the pre-processing step|

<br/>

**Config example** (for Epilepsy):
```
run:
    tissue: Brain
    additional_tissues: []
    seed_include_terms: [epilep, seizure]
    exclude_terms: []
    additional_include_terms: [nerve, nervous, neuronal, cerebellum, cerebral, hippocampus, hypothalamus]
    phenotype: Epilepsy 
    run_id: production
pu_params:
    classifiers: [ExtraTreesClassifier, RandomForestClassifier]
    iterations: 2
    nthreads: 4
run_steps:
    run_boruta: False
    run_unsupervised: True
```

#### Advanced parameters
All other config parameters (_Advanced_) can be used with their default values (see `mantis_ml/conf/config.yaml` template).
<br/><br/>



### 2. Preview selected features based on input config parameters (Optional)
----------------------------------------------------------
```
cd mantis_ml/bin
python mantis_ml_profiler.py input_config.yaml [-v]             # use -v for verbose output
```
Example output available at: `mantis_ml/bin/logs/profiling.out`


<br/><br/>

### 3. Run `mantis-ml`
---------------
```
cd mantis_ml/bin
./run_mantis_ml.sh [-c CONFIG_FILE] 
```

#### Examples
```
cd mantis_ml/bin
./run_mantis_ml.sh -c ../conf/config.yaml
```

<br>

### 3'. Run on a `SLURM` cluster
------------------------
```
cd mantis_ml/bin
sbatch [SBATCH_OPTIONS, e.g. -o, -t] ./submit_mantis_ml.sh [-h] [-c|--config CONFIG_FILE] [-m|--mem MEMORY]
      			 [-t|--time TIME] [-n|--nthreads NUM_THREADS]
```

#### Examples
```
# generic
sbatch -o generic.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh -c ../conf/Generic_config.yaml -m 12G -n 10

# CKD
sbatch -o ckd.sbatch.out -t 24:0:0 ./submit_mantis_ml.sh -c ../conf/CKD_config.yaml
```

<br/><br/>


Output
======
The output folder's name is: `out/[phenotype]/[run_id]`, where `phenotype` and `run_id` are the respective parameters in the input config file.

<br/>

`mantis-ml` predictions (__gene prediction probabilities__ and __percentile scores__) can be found at:
- `out/[OUTPUT_FOLDER]/supervised-learning/ranked_gene_predictions`, 
<br/>

in the **[CLASSIFIER].All_genes.mantis-ml_percentiles.csv** files.



#### Example mantis-ml scores output
```
out/CKD-production/supervised-learning/ranked_gene_predictions/ExtraTreesClassifier.All_genes.mantis-ml_percentiles.csv
```

|Gene_Name|mantis_ml_proba|mantis_ml_perc|
|---------|---------------|--------------|
|MARK2|0.9510298564038976|100.0|
|MAPK8IP3|0.9404119640423988|99.99463116074305|
|CDH2|0.9342689255189254|99.9892623214861|
|PRKG1|0.9156746031746031|99.98389348222915|
|HES1|0.9133577533577534|99.97852464297219|
|CHRNA1|0.906381187440011|99.97315580371524|
|TFRC|0.9061122637525076|99.96778696445828|
|DDB1|0.8967073370255295|99.96241812520134|
| ... | ... | ... |


All generated **figures** can be found at:
```
out/[OUTPUT_FOLDER]/figs
``` 



<br/><br/>


Workflow overview
=================
`mantis-ml` follows and Automated Machine Learning (AutoML) approach for feature extraction (relevant to the disease of interest), feature compilation and pre-processing. The processed feature table is then fed to the main algorithm that lies in the core of `mantis-ml`: a stochastic semi-supervised approach that ranks genes based on their average performance in out-of-bag sets across random balanced datasets from the entire gene pool. 

A set of standard classifiers are used as part of the semi-supervised learning task:
- Random Forests (RF)
- Extremely Randomised Trees (Extra Trees; ET)
- Gradien Boosting (GB)
- Extreme Gradient Boosting (XGBoost)
- Support Vector Classifier (SVC)
- Deep Neural Net (DNN)
- Stacking (1st layer: RF, ET, GB, SVC; 2nd layer: DNN)

The parameters from each classifier have been fine-tuned based on grid search using a subset of balanced datasets constructed on the basis of Chronic Kidney Disease example.

Following the semi-supervised learning step of `mantis-ml`, predictions are overlapped with results from cohort-level rare-variant association studies.

The final consensus results can then been visualised using a set of dimensionality reduction techniques:
- PCA
- t-distributed Stochastic Neighbor Embedding (t-SNE)
- Uniform Manifold Approximation and Projection

<br><br>

Package contents
================
The `mantis-ml-release` project contains:
- the `mantis-ml` package which includes:
  - the `mantis-ml` main run scripts
  <pre>
	.
	└──mantis_ml
		└── bin
  </pre>

  - example input *config* files
  <pre>
	.
	└──mantis_ml
		└── conf
  </pre>

  - the `mantis-ml` modules
  <pre>
	  .  
	  └──mantis_ml
	  	└── modules
			├── post_processing
			├── pre_processing
			│   └── data_compilation
			├── supervised_learn
			│   ├── classifiers
			│   ├── core
			│   ├── feature_selection
			│   ├── model_selection
			│   └── pu_learn
			├── unsupervised_learn
			└── validation
  </pre>
			
  - integrated feature data (`mantis-ml/data/`)
  <pre>
	.
	└──mantis_ml        
		└── data
		    ├── adipose_eqtl
		    ├── ckddb
		    ├── ensembl
		    ├── essential_genes_for_mouse
		    ├── exac-broadinstitute
		    │   └── cnv
		    ├── exSNP
		    ├── genic-intolerance
		    ├── gnomad
		    ├── goa
		    ├── gtex
		    │   └── RNASeq
		    ├── gwas_catalog
		    ├── HPO
		    ├── human_protein_atlas
		    ├── in_web
		    ├── mgi
		    ├── msigdb
		    │   └── tables_per_gene_set
		    ├── neph_qtl
		    ├── omim
		    ├── platelets_eqtl
		    └── rvis_plosgen_2013
  </pre>
		 
- output structure examples (`out/`)
 <pre>
   .
   └── out
	└── CKD-production
		├── data
		│   └── compiled_feature_tables
		├── feature_selection
		│   └── boruta
		│       ├── out
		│       └── tmp
		├── figs
		│   ├── benchmarking
		│   ├── boruta
		│   ├── EDA
		│   ├── supervised-learning
		│   │   └── gene_proba_predictions
		│   └── unsupervised-learning
		├── processed-feature-tables
		├── supervised-learning
		│   ├── gene_predictions
		│   ├── gene_proba_predictions
		│   └── ranked_gene_predictions
		└── unsupervised-learning
 </pre>


- Additional analyses modules: `misc`
  - module for overlap of `mantis-ml` predictions with *cohort-level association studies*
  <pre>
	    .
	    └──misc
		└── overlap-collapsing-analyses
  </pre>

  - module for estimation of *Boruta* feature importance scores *across different disease examples*
  <pre>
	    .
	    └──misc
		└── boruta-post-processing
  </pre>
