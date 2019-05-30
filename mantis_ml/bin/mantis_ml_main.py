import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys
import pandas as pd
import pickle
from mantis_ml.modules.supervised_learn.pu_learn.pu_learning import PULearning
from mantis_ml.modules.pre_processing.eda_wrapper import EDAWrapper
from mantis_ml.modules.pre_processing.feature_table_compiler import FeatureTableCompiler
from mantis_ml.modules.unsupervised_learn.dimens_reduction_wrapper import DimensReductionWrapper
from mantis_ml.modules.post_processing.process_classifier_results import ProcessClassifierResults
from mantis_ml.modules.post_processing.merge_predictions_from_classifiers import MergePredictionsFromClassifiers
from mantis_ml.modules.supervised_learn.feature_selection.run_boruta import BorutaWrapper
from mantis_ml.config_class import Config


class MantisMl:

    def __init__(self, config_file):
        self.cfg = Config(config_file)

        print('Stochastic iterations:', self.cfg.iterations)
        print('nthreads:', self.cfg.nthreads)

		
    def run(self, clf_id=None, final_level_classifier=None, run_feature_compiler=False, run_eda=False, run_pu=False,
                  run_aggregate_results=False, run_merge_results=False,
                  run_boruta=False, run_unsupervised=False):
        # ========= Compile feature table =========
        if run_feature_compiler:
            feat_compiler = FeatureTableCompiler(self.cfg)
            feat_compiler.run()


        # ========= Run EDA and pre-processing =========
        if run_eda:
            eda_wrapper = EDAWrapper(self.cfg)
            eda_wrapper.run()

        data = pd.read_csv(self.cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')


        # ================== Supervised methods ==================
        # ************ Run PU Learning ************
        if run_pu:
            print('Classifier:', clf_id)
            pu = PULearning(self.cfg, data, clf_id, final_level_classifier)
            pu.run()


        # ************ Process predictions per classifier ************
        if run_aggregate_results:
            aggr_res = ProcessClassifierResults(self.cfg, show_plots=True)
            aggr_res.run()


        # ************ Merge results from all classifiers ************
        if run_merge_results:
            merger = MergePredictionsFromClassifiers(self.cfg)
            merger.run()


        # ************ Run Boruta feature seleciton algorithm ************
        if run_boruta:
            boru_wrapper = BorutaWrapper(self.cfg)
            boru_wrapper.run()


        # ========= Unsupervised methods =========
        # PCA, sparse PCA and t-SNE
        if run_unsupervised:
            recalc = False # default: True
        
            if clf_id is None:
                    gene_annot_list = self.cfg.gene_annot_list
            else:
                et_novel_genes = pd.read_csv(str(self.cfg.superv_ranked_pred / (clf_id + '.Novel_genes.Ranked_by_prediction_proba.csv')), header=None, index_col=0)
                print(et_novel_genes.head())
                gene_annot_list = et_novel_genes.head(10).index.values

            dim_reduct_wrapper = DimensReductionWrapper(self.cfg, data, gene_annot_list, recalc)
            dim_reduct_wrapper.run()
			
			
		
    def run_non_clf_specific_analysis(self):
        """
            run_tag: 'pre'
        """
        args_dict = {'run_feature_compiler': True, 'run_eda': True, 'run_unsupervised': self.cfg.run_unsupervised}
        self.run(**args_dict)


    def run_boruta_algorithm(self):
        """
            run_tag: 'boruta'
        """
        args_dict = {'run_boruta': True}
        self.run(**args_dict)

		
    def run_pu_learning(self, clf_id, final_level_classifier):
        """
            run_tag: 'pu'
        """
        args_dict = {'clf_id':clf_id, 'final_level_classifier': final_level_classifier, 'run_pu': True}
        self.run(**args_dict)

		
    def run_post_processing_analysis(self):
        """
            run_tag: 'post'
        """
        args_dict = {'run_aggregate_results': True, 'run_merge_results': True}
        self.run(**args_dict)

		
    def run_clf_specific_unsupervised_analysis(self, clf_id):
        """
            run_tag: 'post_unsup'
        """
        args_dict = {'clf_id': clf_id, 'run_unsupervised': True}
        self.run(**args_dict)
		
    # ----------------------------------------------
    def run_all(self, clf_id):
        """
            run_tag: 'all'
        """
        args_dict = {'clf_id': clf_id, 'run_feature_compiler': True, 'run_eda': True, 'run_pu': True,
                  'run_aggregate_results': True, 'run_merge_results': True,
                  'run_boruta': True, 'run_unsupervised': True}
        self.run(**args_dict)
    # ----------------------------------------------
		


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", dest="config_file",
                        help="config.yaml file with run parameters")
    parser.add_argument("-r", "--run", dest="run_tag", choices=['pre', 'boruta', 'pu', 'post', 'post_unsup', 'all'],
                        help="specify type of analysis to run: pre, boruta, pu, post, post_unsup or all")
    parser.add_argument("-m", "--model", dest="clf_id", choices=['ExtraTreesClassifier', 'DNN', 'RandomForestClassifier', 'SVC', 'XGBoost', 'GradientBoostingClassifier', 'Stacking', 'None'],
                        help="classifier for supervised learning")
    parser.add_argument("-s", "--stacking", dest="final_level_classifier", choices=['DNN', 'None'],
                        help="final level classifier for Stacking")

    args = parser.parse_args()
    print(args)

    config_file = args.config_file
    clf_id = args.clf_id
    run_tag = args.run_tag
    final_level_classifier = args.final_level_classifier

    # TODO: add default arguments

    mantis = MantisMl(config_file)

    if run_tag == 'pre':
        mantis.run_non_clf_specific_analysis()
		
    if run_tag == 'boruta':
        mantis.run_boruta_algorithm()

    if run_tag == 'pu':
        mantis.run_pu_learning(clf_id, final_level_classifier)

    if run_tag == 'post':
        mantis.run_post_processing_analysis()

    if run_tag == 'post_unsup':
        mantis.run_clf_specific_unsupervised_analysis(clf_id)
    
    if run_tag == 'all':
        mantis.run_all(clf_id)
