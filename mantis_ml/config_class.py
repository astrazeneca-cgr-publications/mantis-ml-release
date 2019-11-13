import os, sys
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import yaml
from pathlib import Path
from shutil import copyfile
import pandas as pd
from random import randint


class Config:

    def __init__(self, config_file): 
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        # Read config YAML file
        # self.config_file = Path(self.dir_path + '/' + config_file)
        self.config_file = Path(config_file)  #'config.yaml'

        with open(config_file, 'r') as ymlfile:
            self.conf = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.init_variables()
        self.init_directories()


    def init_variables(self):
        # Specify target variable
        self.Y = self.conf['static']['Y_label']
        self.gene_name = self.conf['static']['gene_name']

        self.phenotype = self.conf['run']['phenotype']
        self.tissue = self.conf['run']['tissue']
        self.additional_tissues = self.conf['run']['additional_tissues']
        self.seed_include_terms = self.conf['run']['seed_include_terms']
        self.exclude_terms = self.conf['run']['exclude_terms']
        self.additional_include_terms = self.conf['run']['additional_include_terms']


        # Run advanced
        # - genes to highlight on plots
        self.gene_annot_list = self.conf['run_advanced']['gene_annot_list']
        self.anchor_genes = self.conf['run_advanced']['anchor_genes']

        self.include_disease_features = self.conf['run_advanced']['include_disease_features']
        self.generic_classifier = self.conf['run_advanced']['generic_classifier']
        if self.generic_classifier == 'None':
            self.generic_classifier = None
        self.hide_seed_genes_ratio = self.conf['run_advanced']['hide_seed_genes_ratio']
        self.seed_pos_ratio = self.conf['run_advanced']['seed_pos_ratio']
        self.random_seeds = self.conf['run_advanced']['random_seeds']

        # PU learning parameters
        self.classifiers = self.conf['pu_params']['classifiers']
        self.iterations = self.conf['pu_params']['iterations']
        self.nthreads = self.conf['pu_params']['nthreads']

        # Data dir with input feature tables to be processed and compiled
        self.data_dir = Path(self.dir_path + '/' + self.conf['static']['data_dir'])

        # Define default gene-set
        self.hgnc_genes_series = pd.read_csv(self.data_dir / 'exac-broadinstitute/all_hgnc_genes.txt', header=None).loc[:, 0]

        ## === DIRS ===
        # Root Output path
        self.out_root = Path(self.dir_path + '/../out/' + self.phenotype + '-' + self.conf['run']['run_id'])
        print(self.out_root)

        # Root Figs output dir
        self.figs_dir = self.out_root / "figs"

        # Output dir to store processed feature tables
        self.processed_data_dir = self.out_root / "processed-feature-tables"

        # Unsupervised learning predictions/output folder
        self.unsuperv_out = self.out_root / 'unsupervised-learning'

        # Supervised learning predictions/output folder
        self.superv_out = self.out_root / 'supervised-learning'
        self.superv_pred = self.superv_out / 'gene_predictions'
        self.superv_ranked_pred = self.superv_out / 'ranked_gene_predictions'
        self.superv_proba_pred = self.superv_out / 'gene_proba_predictions'

        # Output foldr for classifier benchmarking output
        self.benchmark_out = self.figs_dir / 'benchmarking'

        # EDA output folder for figures
        self.eda_out = self.figs_dir / 'EDA'

        # Unsupervised learning figures folder
        self.unsuperv_figs_out = self.figs_dir / 'unsupervised-learning'

        # Supervised learning figures folder
        self.superv_figs_out = self.figs_dir / 'supervised-learning'
        self.superv_figs_gene_proba = self.superv_figs_out / 'gene_proba_predictions'

        # Run steps (remove/add boruta and/or unsupervised steps)
        self.run_boruta = self.conf['run_steps']['run_boruta']
        self.run_unsupervised = self.conf['run_steps']['run_unsupervised']

        # Boruta feature selection output data & figures
        self.boruta_figs_dir = self.figs_dir / 'boruta'
        self.feature_selection_dir = self.out_root / 'feature_selection'
        self.boruta_tables_dir = self.feature_selection_dir / 'boruta'

        # Read filter args
        self.discard_highly_correlated = self.conf['eda_filters']['discard_highly_correlated']
        self.create_plots = self.conf['eda_filters']['create_plots']
        self.drop_missing_data_features = self.conf['eda_filters']['drop_missing_data_features']
        self.drop_gene_len_features = self.conf['eda_filters']['drop_gene_len_features']
        self.manual_feature_selection = self.conf['eda_filters']['manual_feature_selection']

        # Read other parameters for EDA
        self.missing_data_thres = self.conf['eda_parameters']['missing_data_thres']
        self.high_corr_thres = self.conf['eda_parameters']['high_corr_thres']

        # Read parameters for supervised learning
        self.feature_selection = self.conf['supervised_filters']['feature_selection']
        self.boruta_iterations = self.conf['supervised_filters']['boruta_iterations']
        self.boruta_decision_thres = self.conf['supervised_filters']['boruta_decision_thres']
        self.add_original_features_in_stacking = self.conf['supervised_filters']['add_original_features_in_stacking']
        self.test_size = self.conf['supervised_filters']['test_size']
        self.balancing_ratio = self.conf['supervised_filters']['balancing_ratio']
        self.random_fold_split = self.conf['supervised_filters']['random_fold_split']
        self.kfold = self.conf['supervised_filters']['kfold']

        # randomisation
        self.random_state = 2018  # randint(1, 100000000)

        # ============================
        # Dir with compiled feature tables
        self.out_data_dir = Path(self.out_root / 'data')
        self.compiled_data_dir = Path(self.out_data_dir / 'compiled_feature_tables')

        # Define input feature tables
        self.generic_feature_table = Path(self.compiled_data_dir / 'generic_feature_table.tsv')
        self.filtered_by_disease_feature_table = Path(self.compiled_data_dir / 'filtered_by_disease_feature_table.tsv')
        self.ckd_specific_feature_table = Path(self.compiled_data_dir / 'ckd_specific_feature_table.tsv')
        self.cardiov_specific_feature_table = Path(self.compiled_data_dir / 'cardiov_specific_feature_table.tsv')

        self.complete_feature_table = Path(self.compiled_data_dir / 'complete_feature_table.tsv')



    def init_directories(self):
        '''
        - Create output dirs
        - Copy config.yaml to out_root directory
        :return: 
        '''

        if not os.path.exists(self.compiled_data_dir):
            os.makedirs(self.compiled_data_dir)

        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)

        if not os.path.exists(self.out_data_dir):
            os.makedirs(self.out_data_dir)

        if not os.path.exists(self.figs_dir):
            os.makedirs(self.figs_dir)

        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)

        if not os.path.exists(self.eda_out):
            os.makedirs(self.eda_out)

        if not os.path.exists(self.unsuperv_out):
            os.makedirs(self.unsuperv_out)

        if not os.path.exists(self.unsuperv_figs_out):
            os.makedirs(self.unsuperv_figs_out)

        if not os.path.exists(self.superv_out):
            os.makedirs(self.superv_out)

        if not os.path.exists(self.superv_pred):
            os.makedirs(self.superv_pred)

        if not os.path.exists(self.superv_ranked_pred):
            os.makedirs(self.superv_ranked_pred)

        if not os.path.exists(self.superv_proba_pred):
            os.makedirs(self.superv_proba_pred)

        if not os.path.exists(self.superv_figs_out):
            os.makedirs(self.superv_figs_out)

        if not os.path.exists(self.superv_figs_gene_proba):
            os.makedirs(self.superv_figs_gene_proba)

        if not os.path.exists(self.benchmark_out):
            os.makedirs(self.benchmark_out)

        if not os.path.exists(self.boruta_figs_dir):
            os.makedirs(self.boruta_figs_dir)

        if not os.path.exists(self.feature_selection_dir):
            os.makedirs(self.feature_selection_dir)

        if not os.path.exists(self.boruta_tables_dir):
            os.makedirs(self.boruta_tables_dir)

        # Copy input config.yaml to output dir
        src_conf = str(self.config_file)
        dest_conf = str(self.out_root / 'config.yaml')
        copyfile(src_conf, dest_conf)

        self.locked_config_path = dest_conf
        # print('Locked config path:', self.locked_config_path)


if __name__ == '__main__':

    conf_file = sys.argv[1]
    cfg = Config(conf_file)
