# mantis-ml 

- [Introduction](#introduction) 
- [Installation](#installation) 
- [Run](#run) 



Introduction
============
`mantis-ml` is a disease-agnostic gene prioritisation framework, implementing stochastic semi-supervised learning on top of `scikit-learn` and `keras`/`tensorflow`.  
`mantis-ml` takes its name from the Greek word 'μάντης' which means 'fortune teller', 'predicter'.

<br>

|Citation|
|----|
|Vitsios Dimitrios, Petrovski Slavé. **Stochastic semi-supervised learning to prioritise genes from high-throughput genomic screens**. <br/>
https://www.biorxiv.org/content/10.1101/655449v1 bioRxiv, May 30, 2019, [doi:10.1101/655449](https://doi.org/10.1101/655449) |


### Gene prioritisation Atlas:
[https://dvitsios.github.io/mantis-ml-predictions](https://dvitsios.github.io/mantis-ml-predictions)

This resource contains gene prediction results extracted by **mantis-ml** across 10 disease areas in 6 specialties: _Cardiology_, _Immunology_, _Nephrology_, _Neurology_, _Psychiatry_ and _Pulmonology_.


<br>

Installation
============
### Requirements:
- **Python3** (tested with v3.6.7)   [Required]
- **Anaconda3** (tested with v5.3.0) [Recommended]

<br>

**1. Download `mantis-ml-release` GitHub repository:**
```
git clone https://github.com/astrazeneca-cgr-publications/mantis-ml-release.git
```

<br/>

**2. Create a new `conda` environment:** [Recommended]
- [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
- Create a new environment with clean Python and R installations:
```
conda create -n mantis_ml python=3.6 r
conda config --append channels conda-forge   	# add conda-forge in the channels list
conda activate mantis_ml			# activate the newly created conda environment
```

<br>

**3. Install `Python` library dependencies:**
```
python setup.py install
```

You can now call the following scripts from the command line:
- **mantisml**: run mantis-ml gene prioritisation based on a provided config file (.yaml)
- **mantisml-preview**: preview selected phenotypes and features based on a provided config file
- **mantisml-overlap**: run enrichment test between mantis-ml predictions and an external ranked gene list to get refined gene predictions

Run each command with `-h` to see all available options.
<br>

---

### Config file
#### Required parameters:
- `Disease/Phenotype terms`: terms that characterise a phenotype or disease of interest

#### Optional parameters:
- `Additional associated terms`: terms used in addition to `Disease/Phenotype terms` to extract disease/phenotype-associated features 
- `Diseases/Phenotypes to exclude`: terms to exclude from disease/phenotype characterisation and feature selection


**Config example** - (Epilepsy):
```
Disease/Phenotype terms: epileptic, epilepsy, seizure
Additional associated terms: brain, nerve, nervous, neuronal, cerebellum, cerebral, hippocampus, hypothalamus
Diseases/Phenotypes to exclude: 
```

Other example config files can be found under `mantis-ml/conf`. 

---

Run
===
## `mantisml`
You need to provide a config file (.yaml) and an output directory. You may also define the number of threads to use (-n option)
```
mantisml -c [config_file] -o -o [output_dir] [-n nthreads]
```

#### Example
```
mantisml -c CKD_config.yaml -o ./CKD-run
mantisml -c Epilepsy_config.yaml -o /tmp/Epilepsy-testing -n 20
```

<br/><br/>


#### `mantisml` Output
`mantisml` predictions for all genes and across all classifiers can be found at **[output_dir]/Gene-Predictions**. 
The `AUC_performance_by_Classifier.pdf` file under the same dir contains information about the AUC performance per classifier and thus informs about the best performing classifier.

Output figures from all steps during the `mantis-ml` run (e.g. Exploratory Data Analysis/EDA, supervised-learning, unsupervised-learning) can be found under **[output_dir]/Output-Figures**.

<br/>

## `mantisml-profiler`

### Preview selected phenotypes and features (optional)
You may preview all selected features based on your input config file parameters by running the `mantisml-profiler` command.
This allows the user to view which HPO phenotypes and features are picked up based on the given input parameters. Based on the results, the user may further tweak their input config file to better refelct the set of phenotypes and/or features that are more relevant for their disease/case under study.

To run `mantisml-profiler`, you need to provide a config file (.yaml) and an output directory
```
mantisml-profiler [-v] -c [config_file] -o [output_dir]
```

<br/><br/>

## `mantisml-overlap`
### Run enrichment test between mantis-ml predictions and an external ranked gene list to get refined gene predictions

To run `mantisml-overlap`, you need to provide a config file (.yaml), an output directory with mantis-ml output and an external ranked gene list file
```
mantisml-overlap -c [config_file] -o [output_dir] -e [external_ranked_file]
```

#### `mantisml-overlap` Output
Results are available under `[output_dir]/Overlap-Enrichment-Results`.

- `mantisml-overlap` generates figures with the enrichment signal between mantis-ml predictions and the external ranked file, based on a hypergeometric test. These can be found under: `Overlap-Enrichment-Results/hypergeom-enrichment-figures`

- `mantisml-overlap` also extracts consensus gene predictions with support by multiple classifiers. Results can be found at `Overlap-Enrichment-Results/Gene-Predictions-After-Overlap/



#### Advanced Config parameters
All advanced config parameters (**Advanced**) can be used with their default values in `mantis_ml/conf/.config`.
The number of threads and the number of stochastic iterations may be specified by the user during the `mantisml` run with the `-n` and `-i` options respectively.
<br/><br/>
