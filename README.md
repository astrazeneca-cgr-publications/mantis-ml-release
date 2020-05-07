# mantis-ml 


<img style="display:block; margin:0 auto;" src="misc/mantis-ml_logo.png" width="160">

- [Introduction](#introduction) 
- [Installation](#installation) 
- [Run](#run) 
  - [mantisml](#mantisml)
  - [mantisml-profiler](#mantisml-profiler)
  - [mantisml-overlap](#mantisml-overlap)



Introduction
============
`mantis-ml` is a disease-agnostic gene prioritisation framework, implementing stochastic semi-supervised learning on top of `scikit-learn` and `keras`/`tensorflow`.  
`mantis-ml` takes its name from the Greek word 'μάντης' which means 'fortune teller', 'predicter'.

<br>

|Publication - Please cite: |
| :---- |
|[Mantis-ml: Disease-Agnostic Gene Prioritization from High-Throughput Genomic Screens by Stochastic Semi-supervised Learning](https://www.cell.com/ajhg/fulltext/S0002-9297(20)30108-7). <br/>
Dimitrios Vitsios, Slavé Petrovski.  <br/>
__The American Journal of Human Genetics__ (Cell Press), May 7, 2020 https://doi.org/10.1016/j.ajhg.2020.03.012  |

<br>

| Gene prioritisation Atlas: |
| :---- |
| [https://dvitsios.github.io/mantis-ml-predictions](https://dvitsios.github.io/mantis-ml-predictions) |
| This resource contains gene prediction results extracted by **mantis-ml** across **10 disease areas** in **6 specialties**: _Cardiology_, _Immunology_, _Nephrology_, _Neurology_, _Psychiatry_ and _Pulmonology_. |


<br>

Installation
============
**Requirements:** `Python3` (tested with v3.6.7)

`mantis-ml` can be installed through `pip`:
```
pip install mantis-ml
```

<br>

Alternatively, it can be installed from the github repository:

```
git clone https://github.com/astrazeneca-cgr-publications/mantis-ml-release.git
python setup.py install
```

<br>

---

In either case, it is highly recommended to **create a new virtual environment** (e.g. with `conda`) before installing `mantis-ml`:
```
conda create -n mantis_ml python=3.6
conda config --append channels conda-forge   	# add conda-forge in the channels list
conda activate mantis_ml			# activate the newly created conda environment
```

---

<br>


You may now call the following scripts from the command line:
- **`mantisml`**: run mantis-ml gene prioritisation based on a provided config file (`.yaml`)
- **`mantisml-preview`**: preview selected phenotypes and features based on a provided config file
- **`mantisml-overlap`**: run enrichment test between mantis-ml predictions and an external ranked gene list to get refined gene predictions

Run each command with `-h` to see all available options.


<br><br>



Run
===

You need to provide a config file (`.yaml`) containing information about the diseases/phenotypes of interest.
<br>


#### Required field:
- `Disease/Phenotype terms`: **terms that characterise a phenotype or disease of interest** (*free text*)


#### Optional fields:
- `Additional associated terms`: terms used along with `Disease/Phenotype terms` to extract additional disease/phenotype-associated features (*free text*)
- `Diseases/Phenotypes to exclude`: terms to exclude from disease/phenotype characterisation and feature selection (*free text*)


<br>


**Config examples**:
```
# Epilepsy_config.yaml
Disease/Phenotype terms: epileptic, epilepsy, seizure
Additional associated terms: brain, nerve, nervous, neuronal, cerebellum, cerebral, hippocampus, hypothalamus
Diseases/Phenotypes to exclude: 
```
```
# CKD_config.yaml
Disease/Phenotype terms: renal, kidney, nephro, glomerul, distal tubule 
Additional associated terms: 
Diseases/Phenotypes to exclude: adrenal
```

Other example config files can be found under [example-input](example-input) or `mantis-ml/conf`. 

<br>


#### Supervised learning models
- `mantis-ml` runs 6 different supervised models by default: Extra Trees, Random Forest, SVC, Gradient Boosting, XGBoost and Deep Neural Net. 
- It is also possible to run `mantis-ml` with the `-f / --fast` option, which will force mantis-ml to train only 4 classifiers: `Extra Trees`, `Random Forest`, `SVC` and `Gradient Boosting`.
- Additionally, the user may explicitly specify which supervised models to be used for training via the `-m` option. The available model options are coded as follows:
  - `et`: Extra Trees
  - `rf`: Random Forest
  - `gb`: Gradient Boosting
  - `xgb`: XGBoost
  - `svc`: Support Vector Classifier
  - `dnn`: Deep Neural Net
  - `stack`: Stacking classifier

Multiple models may be specified using a `,` separator, e.g. `-m et`, `-m et,stack,gb` etc. 


#### Estimated run time

`mantis-ml` total run time is inversely proportional to the number of known disease-associated (seed) genes (the fewer the seed genes are the more balanced datasets there are to be trained). 
<br>
Example run times for different numbers of seed genes are given in this table. All results correspond to `mantis-ml` runs across **10 stochastic iterations**, training with **6 different supervised models** and using **10 cores**.

| Disease example| Num. of seed genes | Total run time |
| -------------- | ------------------ | --------------- |
| Epilepsy | 864 | 2h | 
| Chronic Kidney Disease | 587 | 2.5h |
| Amyotrophic Lateral Sclerosis | 77 | 11h |

Representative examples of run times when using the `-f / --fast` option, two classifiers with the `-m` option or just the Stacking classifer are also given below (CKD dataset, 10 stochastic iterations, 10 cores):

| Number of models | Total run time |
| -------------- |  --------------- |
| 6 (default) | 2.5h |
| 4 (`-f`) | 43m |
| 2 (`-m et,rf`) | 19m | 
| Stacking (`-m stack`) | 1.5h |


<br><br>


`mantisml`
=========
You need to provide a config file (`.yaml`) and an output directory. 
<br>
You may also:
- define the number of threads to use (`-n` option; default value: 4).
- define the number of stochastic iterations (`-i` option; default value: 10)
- provide a file with custom seed genes (`-k` option; file should contain new-line separated HGNC names; bypasses HPO)

```
mantisml -c [config_file] -o [output_dir] [-n nthreads] [-i iterations] [-k custom_seed_genes.txt]
```

#### Example
```
mantisml -c CKD_config.yaml -o ./CKD-run
mantisml -c Epilepsy_config.yaml -o /tmp/Epilepsy-testing -n 20
```


#### `mantisml` Output
`mantisml` predictions for all genes and across all classifiers can be found at **`[output_dir]/Gene-Predictions`**. 
<br>
The `AUC_performance_by_Classifier.pdf` file under the same dir contains information about the AUC performance per classifier and thus informs about the best performing classifier.

Output figures from all steps during the `mantis-ml` run (e.g. *Exploratory Data Analysis/EDA, supervised-learning, unsupervised-learning*) can be found under **`[output_dir]/Output-Figures`**.

<br>

`mantisml-profiler`
==================

#### Preview selected phenotypes and features (optional)
You may preview all selected phenotypes and relevant features based on your input config file parameters by running the `mantisml-profiler` command.
<br>

To run `mantisml-profiler`, you need to provide a config file (`.yaml`) and an output directory.
```
mantisml-profiler [-v] -c [config_file] -o [output_dir]
```

<br><br>

`mantisml-overlap`
==================

#### Run enrichment test between mantis-ml predictions and an external ranked gene list to get refined gene predictions

To run `mantisml-overlap`, you need to provide a config file (`.yaml`), an output directory with `mantisml` results and an external ranked gene list file (`mantisml` has to be run already given the same ouput directory).
```
mantisml-overlap -c [config_file] -o [output_dir] -e [external_ranked_file]
```

<br>

#### `mantisml-overlap` external ranked file [-e]
The external ranked gene list file may contain a __single column__ with ranked genes or __two columns__ (tab-delimited), with the 2nd column containing p-values. Examples of external ranked lists for both cases are available at [example-input](example-input).


#### `mantisml-overlap` Output
Results are available under **`[output_dir]/Overlap-Enrichment-Results`**.

- `mantisml-overlap` generates figures with the enrichment signal between mantis-ml predictions and the external ranked file, based on a hypergeometric test. 
These can be found under: **`Overlap-Enrichment-Results/hypergeom-enrichment-figures`**.

- `mantisml-overlap` also extracts consensus gene predictions with support by multiple classifiers. 
Results can be found at **`Overlap-Enrichment-Results/Gene-Predictions-After-Overlap`**.


