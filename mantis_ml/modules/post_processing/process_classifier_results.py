import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from palettable.colorbrewer.sequential import Greens_9
from palettable.colorbrewer.qualitative import Paired_12
from matplotlib.patches import Patch
import seaborn as sns
import sys, os

from mantis_ml.config_class import Config
from mantis_ml.modules.post_processing.evaluate_classifier_predictions import ClassifierEvaluator

feature_imp_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'XGBoost']



class ProcessClassifierResults():
	
	def __init__(self, cfg, show_plots=True):
		self.cfg = cfg
		self.show_plots = show_plots
		self.gene_colors = {'Known': '#bdbdbd', 'Novel': '#31a354', 'Highlighted': '#ef3b2c'}



	def aggregate_classifier_results(self, classifiers, standard_classifiers=True):
		all_clf = dict()
	
		for clf_id in classifiers:
			print('\n' + clf_id)
			clf_eval = ClassifierEvaluator(self.cfg, clf_id)
	
			if clf_id in feature_imp_classifiers:
				clf_eval.plot_avg_feature_imp()
				clf_eval.plot_feat_imp_distribustion()
	
			if standard_classifiers:
				clf_eval.plot_evaluation_metrics()
				clf_eval.get_definitive_gene_predictions(pos_ratio_thres=0.5)

			clf_eval.process_gene_proba_predictions(top_hits=50, make_plots=True)
	
			all_clf[clf_id] = clf_eval

		return all_clf
	
	

	def plot_auc_boxplots_across_all_clf(self, all_clf):
		all_auc_dfs = pd.DataFrame()
		for clf_id, clf in all_clf.items():
			all_auc_dfs[clf_id] = clf.eval_metrics_df['AUC']
	
		#print(all_auc_dfs.head())
	
		# Sort classifiers by AUC average
		class_auc_averages = {}
		for clf_id in all_auc_dfs.keys():
			print(clf_id)
			class_auc_averages[clf_id] = all_auc_dfs[clf_id].mean()
	
		print(class_auc_averages)

		# Write avg. AUC scores per classifier
		out_avg_auc_file = str(self.cfg.superv_out / 'Avg_AUC_per_classifier.txt')
		tmp_fh = open(out_avg_auc_file, 'w')
		for k in sorted(class_auc_averages, key=class_auc_averages.get, reverse=True):
			tmp_fh.write(k + '\t' + str(class_auc_averages[k]) + '\n')
		tmp_fh.close()

	
		# Plot Boxplot
		boxplot_pallete = Greens_9.hex_colors + ['#525252', '#252525', '#000000']
	
		fig, ax = plt.subplots(figsize=(20, 10))
		position = 0
		for clf_id in sorted(class_auc_averages, key=class_auc_averages.get, reverse=False):
			print(clf_id, "Average AUC:", class_auc_averages[clf_id])
	
			cur_feature_series = all_auc_dfs[clf_id]
			cur_feature_series = cur_feature_series[~cur_feature_series.isnull()]
	
			bp = ax.boxplot(cur_feature_series, positions=[position], patch_artist=True, notch=True, widths=0.4,
							flierprops=dict(marker='o', markerfacecolor='black', markersize=3, linestyle='dotted'))
	
			cur_face_color = boxplot_pallete[position]
	
			for patch in bp['boxes']:
				_ = patch.set(facecolor=cur_face_color, alpha=0.9)
	
			position += 1
	
		xtick_labels = []
		for clf_id in sorted(class_auc_averages, key=class_auc_averages.get, reverse=False):
			xtick_labels.append(clf_id + '\n(Avg. AUC: ' + str(round(all_clf[clf_id].avg_auc, 3)) + ')')
	
		_ = ax.set_title('AUC scores from all classifiers in: ' + self.cfg.phenotype)
		_ = ax.set_xticks(range(position + 1))
		_ = ax.set_xticklabels(xtick_labels, rotation=90)
		_ = ax.set_xlim(left=-0.5)
		_ = ax.set_xlabel('Classifiers')
		_ = ax.set_ylabel('AUC score distribution across all runs')
		if self.show_plots:
		   plt.show()
	
		fig.savefig(str(self.cfg.superv_figs_out / 'All_classifiers.AUC_distribution_boxplots.pdf'), bbox_inches='tight')
		fig.savefig(str(self.cfg.superv_ranked_pred / 'AUC_performance_by_Classifier.pdf'), bbox_inches='tight')

	
	
	def plot_gene_counts_per_clf(self, all_clf):
		all_counts_df = pd.DataFrame()
		for clf_id, clf_eval in all_clf.items():
			print(clf_id)
			cur_known_count = len(clf_eval.predicted_known_genes)
			cur_novel_count = len(clf_eval.predicted_novel_genes)
	
			new_row = pd.DataFrame({'Known': cur_known_count, 'Novel': cur_novel_count}, index=[clf_id])
			if len(all_counts_df) > 0:
				all_counts_df = pd.concat([all_counts_df, new_row])
			else:
				all_counts_df = new_row
	
		all_counts_df.sort_values(by='Known', ascending=False, inplace=True)
		#print(all_counts_df.head())
		print(all_counts_df.shape)
	
	
		ax = all_counts_df.plot(kind='bar', stacked=True, figsize=(15, 10), color=[self.gene_colors['Known'], self.gene_colors['Novel']])
		ax.legend(bbox_to_anchor=(1.0, 0.5), fontsize=18)
	
		_ = ax.set_title('Predicted Gene Counts by Classifier', fontsize=22)
		_ = ax.set_xlabel('Classifier', fontsize=18)
		_ = ax.set_ylabel('Gene Counts', fontsize=18)
	
		ax.get_figure().savefig(str(self.cfg.superv_figs_out / 'All_classifiers.Predicted_Gene_Counts.pdf'), bbox_inches='tight')
		if self.show_plots:
			plt.show()
	
	
	
	def get_density_and_cdf_plots(self, all_clf, proba_means, gene_class='Known'):
	
		density_and_cdf_dir = str(self.cfg.superv_figs_out / 'misc')
		if not os.path.exists(density_and_cdf_dir):
			os.makedirs(density_and_cdf_dir)
		
		boxplot_pallete = Paired_12.hex_colors
		plot_cnt = 0
	
		fig, ax = plt.subplots(figsize=(15, 15))
		for clf_id, clf_eval in all_clf.items():
	
			linewidth = 1
			if clf_id in ['ExtraTreesClassifier', 'RandomForestClassifier']:
				linewidth = 3
	
			# Density plot
			sns.distplot(getattr(clf_eval, proba_means), hist=False, kde=True,
						 bins=int(180 / 5), color=boxplot_pallete[plot_cnt],
						 label=clf_id,
						 hist_kws={'edgecolor': 'black'},
						 kde_kws={'shade': True, 'linewidth': linewidth})
	
			plot_cnt += 1
	
		plt.legend(fontsize=18, markerscale=2)
		plt.title('Density plots of ' + gene_class + ' genes proba', fontsize=24)
		plt.xlabel('Prediction probability', fontsize=22)
		plt.ylabel('Density', fontsize=22)
		if self.show_plots:
			plt.show()
		fig.savefig(density_and_cdf_dir + '/All_classifiers.' + gene_class + '_genes_proba_Density_plots.pdf', bbox_inches='tight')
	
		fig, ax = plt.subplots(figsize=(15, 15))
		for clf_id, clf_eval in all_clf.items():
	
			cur_mean_vals = getattr(clf_eval, proba_means).values
			sorted_gene_means = np.sort(cur_mean_vals)
	
			padded_sorted_genes = np.append([0], np.sort(sorted_gene_means))
			padded_sorted_genes = np.append(padded_sorted_genes, [1])
			linestyle = 'solid'
			linewidth = 2
			if clf_id in ['ExtraTreesClassifier', 'RandomForestClassifier']:
				linewidth = 3
			if clf_id in ['Stacking_XGBoost', 'Stacking_DNN']:
				linestyle = 'dashed'
				linewidth = 1
	
			plt.plot(padded_sorted_genes, np.linspace(0, 1, len(sorted_gene_means)+2, endpoint=True), linestyle=linestyle, linewidth=linewidth)
	
			plot_cnt += 1
	
		plt.axvline(x=0.5, linestyle='--', linewidth=2, color='#bdbdbd')
		plt.legend(self.cfg.classifiers, loc='upper left', fontsize=18, markerscale=2)
		plt.title('ECDFs for Prediction probabilities of ' + gene_class + ' genes', fontsize=24)
		plt.xlabel('Prediction probability (p)', fontsize=22)
		plt.ylabel('ECDF(p)', fontsize=22)
		if self.show_plots:
			plt.show()
		fig.savefig(density_and_cdf_dir + '/All_classifiers.' + gene_class + '_genes_proba_CDFs.pdf', bbox_inches='tight')
	
	
	
	def process_merged_clf_proba(self):
	
		merged_clf_eval = ClassifierEvaluator(self.cfg, 'AllClassifiers.Merged')
		merged_clf_eval.process_gene_proba_predictions(top_hits=50)
		#print(merged_clf_eval.gene_proba_df.head())

		return merged_clf_eval
	
	
	def get_correlation_between_classifiers(self, all_clf):
	
		gene_proba_means_per_clf = pd.DataFrame()
		for clf_id in all_clf.keys():
			cur_clf = all_clf[clf_id]
			cur_proba_means_df = all_clf[clf_id].gene_proba_means.to_frame()
			cur_proba_means_df.reset_index(drop=False, inplace=True)
	
			clf_id = clf_id.replace('Classifier', '')
			clf_id = clf_id.replace('Classifiers', '')
			cur_proba_means_df.columns = ['Gene_Name', clf_id]
			#print(cur_proba_means_df.head())
	
			if len(gene_proba_means_per_clf) > 0:
				gene_proba_means_per_clf = pd.merge(gene_proba_means_per_clf, cur_proba_means_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
			else:
				gene_proba_means_per_clf = cur_proba_means_df
	
		#print(gene_proba_means_per_clf.head())

	

	def run(self):
		all_clf = self.aggregate_classifier_results(self.cfg.classifiers)
		self.plot_auc_boxplots_across_all_clf(all_clf)

		print("Saving results to all_clf.pkl")
		with open(str(self.cfg.superv_out / 'all_clf.pkl'), 'wb') as output:
			pickle.dump(all_clf, output, pickle.HIGHEST_PROTOCOL)
		self.get_correlation_between_classifiers(all_clf)





if __name__ == '__main__':

	config_file = sys.argv[1]  #'../../config.yaml'
	cfg = Config(config_file)
	print(cfg.out_root)

	# Create objects to process results from each classifier and merge them
	aggr_res = ProcessClassifierResults(cfg, )

	highlighted_genes = cfg.highlighted_genes


	standard_classifiers = True
	recalc = True

	# ==== For Development purposes - to speed up testing
	all_clf_filepath = str(cfg.superv_out / 'all_clf.pkl')
	if os.path.exists(all_clf_filepath) and not recalc:
		print("Reading all_clf.pkl")
		with open(all_clf_filepath, 'rb') as input:
			all_clf = pickle.load(input)
		print(all_clf.keys())
	else:
		print("Saving results to all_clf.pkl")
		all_clf = aggr_res.aggregate_classifier_results(cfg.classifiers, standard_classifiers=standard_classifiers)
		if standard_classifiers:
			aggr_res.plot_auc_boxplots_across_all_clf(all_clf)

		with open(all_clf_filepath, 'wb') as output:
			pickle.dump(all_clf, output, pickle.HIGHEST_PROTOCOL)

		aggr_res.get_correlation_between_classifiers(all_clf)
