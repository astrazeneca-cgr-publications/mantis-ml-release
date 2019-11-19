- [Advanced Config parameters](#advanced-config-parameters) 
- [Workflow overview](#workflow-overview) 
- [Package contents](#package-contents) 



Advanced Config parameters
==========================
All advanced config parameters (**Advanced**) can be used with their default values in `mantis_ml/conf/.config`.
The number of threads and the number of stochastic iterations may be specified by the user during the `mantisml` run with the `-n` and `-i` options respectively.
<br><br>



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
