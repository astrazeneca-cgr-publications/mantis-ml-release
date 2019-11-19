import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from palettable.colorbrewer.sequential import Greens_9
from palettable.colorbrewer.qualitative import Paired_12
from matplotlib.patches import Patch
import seaborn as sns
from collections import Counter
import sys
from scipy.stats import pearsonr

from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.ml_plot_functions import plot_feature_imp_for_classifier

# classifiers = ['DNN', 'ExtraTreesClassifier', 'RandomForestClassifier'] #, 'Stacking_XGBoost', 'Stacking_DNN', 'SVC', 'XGBoost', 'GradientBoostingClassifier'] #'GradientBoostingClassifier': params need tweaking
# if self.cfg.generic_classifier:
#	 classifiers = ['ExtraTreesClassifier']
feature_imp_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'XGBoost']
gene_colors = {'Known': '#bdbdbd', 'Novel': '#31a354', 'Highlighted': 'red'}


class ClassifierEvaluator:
	def __init__(self, cfg, clf_id):
		self.cfg = cfg
		self.clf_id = clf_id

		try:
			proc_feat_df = pd.read_csv(self.cfg.processed_data_dir / 'processed_feature_table.tsv', sep='\t')
			self.ordered_features = [f for f in proc_feat_df.columns.values if f not in ['known_gene', 'Gene_Name']]

			tmp_eval_metrics_df = pd.read_csv(self.cfg.superv_out / ('PU_' + self.clf_id + '.evaluation_metrics.tsv'), sep='\t')
			self.total_runs = tmp_eval_metrics_df.shape[0]
			del tmp_eval_metrics_df
		except:
			print('Non defined classifier: Instantiated only basic ClassifierEvaluator skeleton.')

		self.known_genes_df = proc_feat_df[['Gene_Name', 'known_gene']]
		self.known_genes = self.known_genes_df.loc[self.known_genes_df.known_gene == 1, 'Gene_Name'].values


	def plot_avg_feature_imp(self):
		avg_feat_imp_df = pd.read_csv(self.cfg.superv_out / ('PU_' + self.clf_id + '.avg_feature_importance.tsv'), sep='\t')
		avg_feat_imp_df.sort_values(by=[self.clf_id], ascending=True, inplace=True)
		#print(avg_feat_imp_df.head())

		plot_feature_imp_for_classifier(avg_feat_imp_df, self.clf_id, self.clf_id, self.cfg.superv_feat_imp)


	def plot_feat_imp_distribustion(self):
		feat_imp_distr_df = pd.read_csv(self.cfg.superv_out / ('PU_' + self.clf_id + '.feature_dfs_list.txt'), sep=',')
		feat_imp_distr_df.columns.values[0] = 'feature_id'
		feat_imp_distr_df.dropna(inplace=True)
		feat_imp_distr_df.reset_index(drop=True, inplace=True)

		feat_imp_distr_df['run_group'] = (feat_imp_distr_df.index.values) / len(self.ordered_features)
		feat_imp_distr_df['run_group'] = feat_imp_distr_df['run_group'].apply(lambda x: int(x))

		pivot_feat_df = feat_imp_distr_df.pivot(index='run_group', columns='feature_id', values=self.clf_id)
		pivot_feat_df.columns = self.ordered_features
		pivot_feat_df.reset_index(drop=True, inplace=True)
		pivot_feat_df = pivot_feat_df.apply(pd.to_numeric, errors='coerce')
		pivot_feat_df = pivot_feat_df.reindex(pivot_feat_df.mean().sort_values().index, axis=1)

		fig, ax = plt.subplots(figsize=(20, 10))

		for position, name in enumerate(pivot_feat_df.columns.values):

			cur_feature_series = pivot_feat_df[name]
			cur_feature_series = cur_feature_series[~cur_feature_series.isnull()]

			bp = ax.boxplot(cur_feature_series, positions=[position], patch_artist=True, notch=True, widths=0.6,
							flierprops=dict(marker='o', markerfacecolor='#bdbdbd', markersize=1,
											linestyle='dotted'))

			cur_face_color = '#3182bd'
			for element in ['boxes', 'whiskers', 'fliers', 'caps']:
				_ = plt.setp(bp[element], color=cur_face_color)
			_ = plt.setp(bp['medians'], color='#737373')

			for patch in bp['boxes']:
				_ = patch.set(facecolor=cur_face_color, alpha=0.9)

		_ = ax.set_title(self.clf_id + ' feature importance in: ' + self.cfg.phenotype)
		_ = ax.set_xticks(range(position + 1))
		_ = ax.set_xticklabels(pivot_feat_df.columns.values, rotation=90)
		_ = ax.set_xlim(left=-0.5)
		_ = ax.set_xlabel('Features')
		_ = ax.set_ylabel('Feature Importance scores Distribution')
		# plt.show()

		fig.savefig(str(self.cfg.superv_feat_imp / (self.clf_id + '_feature_imp_distr_boxplots.pdf')), bbox_inches='tight')
		plt.close()



	def plot_evaluation_metrics(self, make_plots=False):
		self.eval_metrics_df = pd.read_csv(self.cfg.superv_out / ('PU_' + self.clf_id + '.evaluation_metrics.tsv'), sep='\t')
		#print(self.eval_metrics_df.head())

		self.avg_auc = self.eval_metrics_df['AUC'].mean()

		if make_plots:
			fig, ax = plt.subplots(figsize=(5, 10))
			_ = ax.set_title(
				'AUC score distribution across ' + str(self.total_runs) + ' runs\nAverage AUC: ' + str(self.avg_auc))
			_ = ax.set_xlabel(self.clf_id)
			_ = ax.set_ylabel('AUC')
			plt.boxplot(self.eval_metrics_df['AUC'], patch_artist=True, notch=True, widths=0.4,
						flierprops=dict(marker='o', markerfacecolor='black',
										markersize=3, linestyle='dotted'))
			# plt.show()
			ax.get_figure().savefig(str(self.cfg.superv_figs_out / (self.clf_id + '.AUC_boxplot.pdf')), bbox_inches='tight')


	def get_definitive_gene_predictions(self, pos_ratio_thres=0.6):

		gene_pred_df = pd.read_csv(self.cfg.superv_out / ('PU_' + self.clf_id + '.all_genes_predictions.tsv'), sep='\t')
		gene_pred_df.columns.values[0] = 'Gene_Name'
		gene_pred_df = pd.merge(gene_pred_df, self.known_genes_df, left_on='Gene_Name', right_on='Gene_Name',
									 how='outer')

		gene_pred_df['pos_ratio'] = gene_pred_df['positive_genes'] / (
		gene_pred_df['positive_genes'] + gene_pred_df['negative_genes'])
		gene_pred_df.sort_values(by=['pos_ratio'], ascending=False, inplace=True)
		# print(gene_pred_df.head())

		predicted_genes_df = gene_pred_df.loc[gene_pred_df.pos_ratio >= pos_ratio_thres]
		predicted_known_genes_df = gene_pred_df.loc[
			(gene_pred_df.pos_ratio >= pos_ratio_thres) & (gene_pred_df.known_gene == 1)]
		predicted_novel_genes_df = gene_pred_df.loc[
			(gene_pred_df.pos_ratio >= pos_ratio_thres) & (gene_pred_df.known_gene == 0)]

		self.predicted_genes = sorted(predicted_genes_df['Gene_Name'].values)
		self.predicted_known_genes = sorted(predicted_known_genes_df['Gene_Name'].values)
		self.predicted_novel_genes = sorted(predicted_novel_genes_df['Gene_Name'].values)

		print(f"Predicted {self.cfg.phenotype} genes: {len(self.predicted_genes)}")
		print(f"Predicted Known {self.cfg.phenotype} genes: {len(self.predicted_known_genes)}")
		print(f"Predicted Novel {self.cfg.phenotype} genes: {len(self.predicted_novel_genes)}")

		# Write results to files
		with open(str(self.cfg.superv_pred / (self.clf_id + '.predicted_genes.txt')), 'w') as f:
			for gene in self.predicted_genes:
				f.write("%s\n" % gene)

		with open(str(self.cfg.superv_pred / (self.clf_id + '.predicted_known_genes.txt')), 'w') as f:
			for gene in self.predicted_known_genes:
				f.write("%s\n" % gene)

		with open(str(self.cfg.superv_pred / (self.clf_id + '.predicted_novel_genes.txt')), 'w') as f:
			for gene in self.predicted_novel_genes:
				f.write("%s\n" % gene)


	def gene_ranking_boxplot(self, genelist, list_id):

		fig, ax = plt.subplots(figsize=(20, 10))
		top_hits = len(genelist)

		position = 0
		for gene in genelist:

			cur_gene_proba = self.gene_proba_df[gene]
			cur_gene_proba = cur_gene_proba[~cur_gene_proba.isnull()]

			bp = ax.boxplot(cur_gene_proba, positions=[position], patch_artist=True, notch=True, widths=0.4,
							flierprops=dict(marker='o', markerfacecolor='black', markersize=3,
											linestyle='dotted'))

			cur_face_color = gene_colors['Novel']
			if gene in self.known_genes:
				cur_face_color = gene_colors['Known']
			if gene in self.cfg.highlighted_genes:
				cur_face_color = gene_colors['Highlighted']

			# for patch in bp['boxes']:
			#	 _ = patch.set(facecolor=cur_face_color, alpha=0.9)

			for element in ['boxes', 'whiskers', 'fliers', 'caps']:
				_ = plt.setp(bp[element], color=cur_face_color)
				_ = plt.setp(bp['medians'], color='#737373')

			position += 1

		xtick_labels = []
		for gene in genelist:
			xtick_labels.append(gene)

		_ = ax.set_title('[' + list_id + ' genes] - Top ' + str(top_hits) + ' ' + self.cfg.phenotype + ' genes')
		_ = ax.set_xticks(range(position + 1))
		_ = ax.set_xticklabels(xtick_labels, rotation=90)
		_ = ax.set_xlim(left=-0.5)
		_ = ax.set_xlabel('Genes')
		_ = ax.set_ylabel('Prediciton probability distribution')

		custom_legend_elements = [Patch(facecolor=gene_colors['Known'], edgecolor=None, label='Known genes'),
								  Patch(facecolor=gene_colors['Novel'], edgecolor=None, label='Novel genes'),
								  #Patch(facecolor=gene_colors['Highlighted'], edgecolor=None, label='Highlighted genes')
								  ]
		ax.legend(handles=custom_legend_elements, bbox_to_anchor=(1.125, 0.55))
		fig.savefig(str(self.cfg.superv_figs_gene_proba / (
		self.clf_id + '.' + list_id + '_genes.top_' + str(top_hits) + '.pdf')), bbox_inches='tight')
		# plt.show()
		plt.close()


	def process_gene_proba_predictions(self, top_hits=50, save_to_file=False, top_hits_to_save=1000,
									   make_plots=True, pos_decision_thres=0.50):

		# Read dictionary text file with gene prediciton probabilities
		proba_df_file = self.cfg.superv_proba_pred / (self.clf_id + '.all_genes.predicted_proba.h5')
		try:
			self.gene_proba_df = pd.read_hdf(proba_df_file, key='df')
			print('Completed reading of ' + self.clf_id + '.all_genes.predicted_proba.h5')
		except Exception as e:
			print(e, '\nFile Not Found: ' + str(proba_df_file) + '.\nSkipping ' + self.clf_id + ' classifier')
			return -1

		# ===== Boxplot for 'All' genes =====
		all_genes_top_hits = self.gene_proba_df.iloc[:, 0:top_hits].columns.values
		if make_plots:
			self.gene_ranking_boxplot(all_genes_top_hits, 'All')
		self.gene_proba_means = self.gene_proba_df.mean(axis=0)
		self.gene_proba_medians = self.gene_proba_df.median(axis=0)


		self.percentile_df = self.gene_proba_means.to_frame()
		self.percentile_df.reset_index(drop=False, inplace=True)
		self.percentile_df.columns = ['Gene_Name', 'mantis_ml_proba']
		self.percentile_df['mantis_ml_perc'] = 100 * self.percentile_df['mantis_ml_proba'].rank(pct=True)
		#print(self.percentile_df.head())

		self.gene_proba_means.to_csv(self.cfg.superv_ranked_by_proba / (self.clf_id + '.All_genes.Ranked_by_prediction_proba.csv'), header=False)


		# ===== Boxplot for 'Known' genes =====
		self.known_gene_proba_df = self.gene_proba_df[self.known_genes]
		self.known_gene_proba_df = self.known_gene_proba_df.reindex(
			self.known_gene_proba_df.mean().sort_values(ascending=False).index, axis=1)

		if save_to_file:
			self.known_gene_proba_df.to_hdf(
				self.cfg.superv_proba_pred / (self.clf_id + '.known_genes.predicted_proba.h5'))
			print('Completed writing of file', self.clf_id + '.known_genes.predicted_proba.h5')
		if make_plots:
			known_genes_top_hits = self.known_gene_proba_df.iloc[:, 0:top_hits].columns.values
			self.gene_ranking_boxplot(known_genes_top_hits, 'Known')
		self.known_gene_proba_means = self.known_gene_proba_df.mean(axis=0)
		self.known_gene_proba_means.to_csv(self.cfg.superv_ranked_by_proba / (self.clf_id + '.Known_genes.Ranked_by_prediction_proba.csv'), header=False)


		# ===== Boxplot for 'Unlabelled' genes =====
		unlabelled_genes = self.known_genes_df.loc[self.known_genes_df.known_gene == 0, 'Gene_Name'].values

		self.unlabelled_gene_proba_df = self.gene_proba_df[unlabelled_genes]
		self.unlabelled_gene_proba_df = self.unlabelled_gene_proba_df.reindex(
			self.unlabelled_gene_proba_df.mean().sort_values(ascending=False).index, axis=1)

		if save_to_file:
			self.unlabelled_gene_proba_df.to_hdf(
				self.cfg.superv_proba_pred / (self.clf_id + '.unlabelled_genes.predicted_proba.h5'))
			print('Completed writing of file', self.clf_id + '.unlabelled_genes.predicted_proba.h5')
		if make_plots:
			unlabelled_genes_top_hits = self.unlabelled_gene_proba_df.iloc[:, 0:top_hits].columns.values
			self.gene_ranking_boxplot(unlabelled_genes_top_hits, 'Novel')
		self.unlabbeled_gene_proba_means = self.unlabelled_gene_proba_df.mean(axis=0)
		self.unlabbeled_gene_proba_means.to_csv(self.cfg.superv_ranked_by_proba / (self.clf_id + '.Novel_genes.Ranked_by_prediction_proba.csv'), header=False)


		print('known_genes_df:')
		print(self.known_genes_df.head())
		print(self.known_genes_df.shape)

		print('percentile_df:')
		print(self.percentile_df.head())
		print(self.percentile_df.shape)


		# Add known_gene annotation in output results
		self.percentile_df = pd.merge(self.percentile_df, self.known_genes_df, left_on='Gene_Name', right_on='Gene_Name', how='left')
		print('Merged percentile_df:')
		print(self.percentile_df.head())
		print(self.percentile_df.shape)
		# ====== Store all results (proba, percentile score, known/novel gene flag) ======
		self.percentile_df.to_csv(self.cfg.superv_ranked_pred / (self.clf_id + '.mantis-ml_predictions.csv'), index=False)






if __name__ == '__main__':

	config_file = sys.argv[1] #'../../config.yaml'
	cfg = Config(config_file)

	clf_id = 'XGBoost'
	clf_eval = ClassifierEvaluator(cfg, clf_id)

	# if clf_id in feature_imp_classifiers:
	#	 clf_eval.plot_avg_feature_imp()
	#	 clf_eval.plot_feat_imp_distribustion()

	# clf_eval.plot_evaluation_metrics()
	# clf_eval.get_definitive_gene_predictions(pos_ratio_thres=0.99)
	clf_eval.process_gene_proba_predictions(top_hits=50)


	# Calculate correlation between mantis-ml scores when using mean vs median of all probability scores per gene
	#print(clf_eval.gene_proba_means.head())
	#print(clf_eval.gene_proba_medians.head())
	print("Pearson's correlation from mean vs media gene probabilities:")
	mean_vs_median_corr = pearsonr(clf_eval.gene_proba_means, clf_eval.gene_proba_medians)
	print('r:', mean_vs_median_corr[0])
	print('p-value:', mean_vs_median_corr[1])
