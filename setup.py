from distutils.core import setup
import nltk

setup(
    name='mantis_ml',
    version='1.0',
    description='Stochastic semi-supervised learning to prioritise genes from high-throughput genomic screens',
    author='Dimitrios Vitsios',
    author_email='dvitsios@gmail.com',
    license='MIT',
    classifiers=['Programming Language :: Python :: 3'],
    url='https://github.com/AstraZeneca-CGR/mantis-ml-public-release',

    packages=['mantis_ml', 'mantis_ml.bin', 'mantis_ml.modules', 'mantis_ml.modules.validation',
              'mantis_ml.modules.pre_processing', 'mantis_ml.modules.pre_processing.data_compilation',
              'mantis_ml.modules.post_processing', 'mantis_ml.modules.supervised_learn',
              'mantis_ml.modules.supervised_learn.core', 'mantis_ml.modules.supervised_learn.pu_learn',
              'mantis_ml.modules.supervised_learn.classifiers', 'mantis_ml.modules.supervised_learn.model_selection',
              'mantis_ml.modules.supervised_learn.feature_selection', 'mantis_ml.modules.unsupervised_learn'],

    scripts=['mantis_ml/bin/mantis_ml_wrapper.sh', 'mantis_ml/bin/run_mantis_ml.sh', 'mantis_ml/bin/submit_mantis_ml.sh'],

    package_data={'':['*.r', '*.R', '*.sh', 'data/*', 'data/adipose_eqtl/*', 'data/ckddb/*', 'data/ensembl/*', 'data/essential_genes_for_mouse/*', 
		      'data/exac-broadinstitute/*', 'data/exac-broadinstitute/cnv/*', 'data/exSNP/*', 'data/genic-intolerance/*', 'data/gnomad/*', 'data/goa/*',
		      'data/gtex/*', 'data/gtex/RNASeq/*', 'data/gwas_catalog/*', 'data/HPO/*', 'data/human_protein_atlas/*', 'data/in_web/*', 
		      'data/mgi/*', 'data/msigdb/*', 'data/msigdb/tables_per_gene_set/*', 'data/neph_qtl/*', 'data/omim/*', 'data/platelets_eqtl/*', 
		      'data/random_vs_seed-mantis_ml_scores/*', 'data/random_vs_seed-mantis_ml_scores/ALS/*', 'data/random_vs_seed-mantis_ml_scores/CKD/*', 'data/random_vs_seed-mantis_ml_scores/Epilepsy/*', 'data/rvis_plosgen_2013/*']},
    include_package_data=True
)


nltk.download('stopwords')
