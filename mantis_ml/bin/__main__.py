import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys
import glob
import pandas as pd
import ntpath
import pickle
from mantis_ml.modules.supervised_learn.pu_learn.pu_learning import PULearning
from mantis_ml.modules.pre_processing.eda_wrapper import EDAWrapper
from mantis_ml.modules.pre_processing.feature_table_compiler import FeatureTableCompiler
from mantis_ml.modules.unsupervised_learn.dimens_reduction_wrapper import DimensReductionWrapper
from mantis_ml.modules.post_processing.process_classifier_results import ProcessClassifierResults
from mantis_ml.modules.post_processing.merge_predictions_from_classifiers import MergePredictionsFromClassifiers
from mantis_ml.modules.supervised_learn.feature_selection.run_boruta import BorutaWrapper
from mantis_ml.bin.mantis_ml_profiler import MantisMlProfiler
from mantis_ml.config_class import Config


class MantisMl:

	def __init__(self, config_file, nthreads=4, iterations=10, include_stacking=False):
		self.config_file = config_file
		self.cfg = Config(config_file)

		# modify default config paramters when provided with respective parameters
		self.cfg.nthreads = int(nthreads)
		self.cfg.iterations = int(iterations)
		if include_stacking:
			self.cfg.classifiers.append('Stacking')

		print('nthreads:', self.cfg.nthreads)
		print('Stochastic iterations:', self.cfg.iterations)
		print('Classifiers:', self.cfg.classifiers)




	def get_clf_id_with_top_auc(self):

		auc_per_clf = {}

		metric_files = glob.glob(str(self.cfg.superv_out / 'PU_*.evaluation_metrics.tsv'))

		for f in metric_files:
			clf_id = ntpath.basename(f).split('.')[0].replace('PU_', '')

			tmp_df = pd.read_csv(f, sep='\t', index_col=0)
			avg_auc = tmp_df.AUC.median()
			auc_per_clf[clf_id] = avg_auc

		top_clf = max(auc_per_clf, key=auc_per_clf.get)
		print('Top classifier:', top_clf)

		return top_clf




	def run(self, clf_id=None, final_level_classifier='DNN', run_feature_compiler=False, run_eda=False, run_pu=False,
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
			for clf_id in self.cfg.classifiers:
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
			recalc = False # default: False
		
			if clf_id is None:
					highlighted_genes = self.cfg.highlighted_genes
			else:
				top_genes_num = 40
				novel_genes = pd.read_csv(str(self.cfg.superv_ranked_pred / (clf_id + '.Novel_genes.Ranked_by_prediction_proba.csv')), header=None, index_col=0)
				highlighted_genes = novel_genes.head(top_genes_num).index.values

			dim_reduct_wrapper = DimensReductionWrapper(self.cfg, data, highlighted_genes, recalc)
			dim_reduct_wrapper.run()

			
			

		
	def run_non_clf_specific_analysis(self):
		""" run_tag: pre """

		args_dict = {'run_feature_compiler': True, 'run_eda': True, 'run_unsupervised': self.cfg.run_unsupervised}
		self.run(**args_dict)



	def run_boruta_algorithm(self):
		""" run_tag: boruta """

		args_dict = {'run_boruta': True}
		self.run(**args_dict)


		
	def run_pu_learning(self):
		""" run_tag: pu """
		
		args_dict = {'run_pu': True}
		self.run(**args_dict)


		
	def run_post_processing_analysis(self):
		""" run_tag: post """
		
		args_dict = {'run_aggregate_results': True, 'run_merge_results': True}
		self.run(**args_dict)


		
	def run_clf_specific_unsupervised_analysis(self, clf_id):
		""" run_tag: post_unsup """
		
		args_dict = {'clf_id': clf_id, 'run_unsupervised': True}
		self.run(**args_dict)

		
	# ---------------------- Run Full pipeline ------------------------
	def run_all(self):
		""" run_tag: all """

		args_dict = {'run_feature_compiler': True, 'run_eda': True, 'run_pu': True,
				  'run_aggregate_results': True, 'run_merge_results': True,
				  'run_boruta': False, 'run_unsupervised': True}
		self.run(**args_dict)
	# -----------------------------------------------------------------
		



	# ***** hypergeometric enrichemnt test *****
	def run_enrichment_overlap(self):
		pass




def main():

	parser = ArgumentParser()
	parser.add_argument("-c", "--config", dest="config_file", help="config.yaml file with run parameters", required=True)
	parser.add_argument("-r", "--run", dest="run_tag", choices=['profiler', 'pre', 'boruta', 'pu', 'post', 'post_unsup', 'all'], default='all', help="specify type of analysis to run: profiler, pre, boruta, pu, post, post_unsup or all")
	parser.add_argument("-n", "--nthreads", dest="nthreads", default=4, help="number of threads")
	parser.add_argument("-i", "--iterations", dest="iterations", default=10, help="number of stochastic iterations of semi-supervised learning")
	parser.add_argument("-s", "--stacking", action="count", help="include Stacking in set of classifiers")
	parser.add_argument('-v', '--verbosity', action="count", help="print verbose output verbosity (run with -v option)")          




	args = parser.parse_args()
	print(args)

	config_file = args.config_file
	run_tag = args.run_tag
	nthreads = args.nthreads
	iterations = args.iterations
	stacking = bool(args.stacking)
	verbose = bool(args.verbosity)


	mantis = MantisMl(config_file, nthreads=nthreads, iterations=iterations, include_stacking=stacking)


	if run_tag == 'all':
		mantis.run_all()
	elif run_tag == 'pre':
		mantis.run_non_clf_specific_analysis()
	elif run_tag == 'pu':
		mantis.run_pu_learning()
	elif run_tag == 'post':
		mantis.run_post_processing_analysis()
	elif run_tag == 'post_unsup':
		top_clf = mantis.get_clf_id_with_top_auc()
		mantis.run_clf_specific_unsupervised_analysis(top_clf)
	elif run_tag == 'boruta':
		mantis.run_boruta_algorithm()

	elif run_tag == 'profiler':
		# ***** Preview selected features *****
		profiler = MantisMlProfiler(config_file, verbose=verbose)         
		profiler.run_mantis_ml_profiler()

	elif run_tag == 'overlap':
		mantis.run_enrichment_overlap()



if __name__ == '__main__':
	main()
