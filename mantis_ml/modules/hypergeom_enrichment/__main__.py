import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.stats import hypergeom 
import pandas as pd
import numpy as np
import pickle
from argparse import RawTextHelpFormatter
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 10)
import sys, os
import random
from pathlib import Path
from scipy.integrate import simps

from mantis_ml.config_class import Config
from mantis_ml.modules.hypergeom_enrichment.consensus_predictions import Consensus_Gene_Predictions


clf_alias = {'ExtraTreesClassifier': 'ET', 'SVC': 'SVC', 'DNN': 'DNN', 'RandomForestClassifier': 'RF',
			 'XGBoost': 'XGB', 'GradientBoostingClassifier': 'GB', 'Stacking': 'Stacking'}




class ExternalRankingOverlap:

	def __init__(self, cfg, clf_str, seed_genes, top_ratio=0.05, max_overlapping_genes=50, show_full_xaxis=False, ylim=None):
		self.cfg = cfg
		self.clf_str = clf_str
		self.seed_genes = seed_genes
		self.top_ratio = top_ratio
		self.max_overlapping_genes = max_overlapping_genes
		self.ylim = ylim
		self.show_full_xaxis = show_full_xaxis

		print('top_ratio:', self.top_ratio)
		print('max_overlapping_genes:', self.max_overlapping_genes)
		print('show_full_xaxis:', self.show_full_xaxis)

		self.has_p_values = False

		self.base_enrichment_dir = str(self.cfg.hypergeom_figs_out)
		if not os.path.exists(self.base_enrichment_dir):
			os.makedirs(self.base_enrichment_dir)
		
		self.enrichment_dir = self.base_enrichment_dir + '/' + self.clf_str
		if not os.path.exists(self.enrichment_dir):
			os.makedirs(self.enrichment_dir)



	def read_external_ranked_gene_list(self, external_ranked_file):
		"""
		    Read external file with independent gene ranking [external_ranked_file].
		    This file should ideally contain an exome-wide ranking, e.g. based on p-values
		    extracted from independent genetic association studies or CRISPR sreens.
		    The greater the size of the external list, the more statistical power can be
		    obtained by the hypergeometric enrichment test between mantis-ml's predictions
		    and the external independent ranking.
		"""

		#self.external_ranked_df = pd.read_csv(external_ranked_file, sep=',|\t', 
		#				      header=None, engine='python')
		self.external_ranked_df = pd.read_csv(external_ranked_file, sep='\t', 
						      header=None)
		print(self.external_ranked_df.head())

		if self.external_ranked_df.shape[1] > 1:
			self.external_ranked_df.columns = ['Gene_Name', 'p-val']
			self.has_p_values = True

			# Sanity check for p-value column
			if any(self.external_ranked_df['p-val'] < 0) or any(self.external_ranked_df['p-val'] > 1) :
				sys.exit('[Error] The 2nd column of the provided file contains invalid values.\n        Please make sure the 2nd column contains p-value scores.') 
		
		else:
			self.external_ranked_df.columns = ['Gene_Name']

		self.external_ranked_df['external_rank'] = range(1, self.external_ranked_df.shape[0]+1)		


		

	def calc_phred_score(self, pval):
		return -10 * np.log10(pval)



	def find_last_signif_gene_index(self, df, collapsing_signif_thres=0.05):
		for i in range(df.shape[0]):
			if df.loc[i, 'p-val'] > collapsing_signif_thres:
				return i



	def calc_stepwise_hypergeometric(self, all_clf, pval_cutoff=0.05, collapsing_top_ratio=-1, 
					 genes_to_remove=[]):

		# *** Initialisation ***
		fig, ax = plt.subplots(figsize=(18, 13))


		signif_thres = 0.05
		signif_thres = self.calc_phred_score(signif_thres)
		ax.axhline(y=signif_thres, linestyle='--', color='red', label='p-val: 0.05')
		max_x_lim = -1
		# **********************
	

		M = self.external_ranked_df.shape[0]
		print('Population Size:', M)

		hypergeom_pvals = []
		hypergeom_ordered_genes = []


		clf = all_clf[self.clf_str]
		proba_df = clf.gene_proba_df
		proba_df = proba_df.iloc[:, ~proba_df.columns.isin(genes_to_remove)]
		print(proba_df.head())
		print(proba_df.shape)


		# Subset top 'top_ratio' % of mantis-ml predictions to overlap with collapsing results
		proba_df = proba_df.iloc[:, 0:int(self.top_ratio * proba_df.shape[1])]
		mantis_ml_top_genes = list(proba_df.columns.values)
		print(mantis_ml_top_genes[:10])
		print('mantis-ml top genes:', len(mantis_ml_top_genes))



		self.external_ranked_df = self.external_ranked_df.loc[self.external_ranked_df['Gene_Name'].isin(mantis_ml_top_genes)]
		self.external_ranked_df.reset_index(drop=True, inplace=True)


		n = self.external_ranked_df.shape[0]
		print('Total number of Successes:', n)
		print(self.external_ranked_df.head())


		# ************* Hypergeometric Test *************
		for x in range(self.external_ranked_df.shape[0]):
			N = self.external_ranked_df.iloc[x, self.external_ranked_df.shape[1]-1]
			cur_pval = hypergeom.sf(x - 1, M, n, N)

			hypergeom_pvals = hypergeom_pvals + [cur_pval]

			cur_gene = self.external_ranked_df.loc[x, 'Gene_Name']
			hypergeom_ordered_genes = hypergeom_ordered_genes + [cur_gene]
		# ***********************************************

		min_pval = min(hypergeom_pvals)
		hypergeom_pvals = [self.calc_phred_score(pval) for pval in hypergeom_pvals]




		# ------------------------- Start plotting ----------------------------
		linewidth = 1
		ax.plot(hypergeom_pvals, color="#33a02c",
					label=self.clf_str,
					linewidth=linewidth)


		if self.has_p_values:
			last_signif_index = self.find_last_signif_gene_index(self.external_ranked_df)
			if last_signif_index is None:
				last_signif_index = self.external_ranked_df.shape[0]		
		else:
			last_signif_index = self.max_overlapping_genes

		print('last_signif_index:', last_signif_index)
		if last_signif_index > max_x_lim:
			max_x_lim = last_signif_index



		ax.set_xlim(left=-0.5)
		ax.set_xlabel('Top ' + str(round(100 * self.top_ratio, 1)) + '% mantis-ml predicted genes', fontsize=14)
		ax.axvline(x=last_signif_index, linestyle='--', linewidth=1, color="#33a02c")


		y_label = 'Phred score from hypergeometric test\n(significance increasing in positive direction)'
		ax.set_ylabel(y_label, fontsize=14)
		print('Min. enrichment p-value from hypergeometric test:', str(min_pval))




		# ====================================================================
		top_overlapping_genes = hypergeom_ordered_genes[:last_signif_index]
		mantis_ml_top_overlap_genes_ranking = [mantis_ml_top_genes.index(gene)+1 for gene in top_overlapping_genes]

		mantis_ml_top_genes_proba = clf.gene_proba_means.loc[top_overlapping_genes]



		merged_results_df = pd.DataFrame({'Gene_Name': top_overlapping_genes, 'mantis_ml_rank': mantis_ml_top_overlap_genes_ranking})
		merged_results_df = pd.merge(merged_results_df, clf.percentile_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
		merged_results_df = pd.merge(merged_results_df, self.external_ranked_df, how='left', left_on='Gene_Name', right_on='Gene_Name')

		merged_results_df = pd.merge(merged_results_df, pd.DataFrame({'Gene_Name': self.seed_genes, 'Known_gene': 1}),
									 how='left', left_on='Gene_Name', right_on='Gene_Name')
		merged_results_df.fillna(0, inplace=True)


		merged_results_df.sort_values(by='mantis_ml_rank', ascending=True, inplace=True)
		merged_results_df.to_csv(self.enrichment_dir + '/mantis_ml-vs-external_ranked_list.Top_' + str(self.top_ratio) + '.' + clf_alias[self.clf_str] + '.csv', index=False)

		novel_overlaping_genes = merged_results_df.loc[ merged_results_df.Known_gene == 0, 'Gene_Name']
		novel_overlaping_genes.to_csv(self.enrichment_dir + '/Novel_overlaping_genes.Top_' + str(self.top_ratio) + '.' + clf_alias[self.clf_str] + '.csv', index=False, header=False)

		known_overlaping_genes = merged_results_df.loc[ merged_results_df.Known_gene == 1, 'Gene_Name']
		known_overlaping_genes.to_csv(self.enrichment_dir + '/Known_overlaping_genes.Top_' + str(self.top_ratio) + '.' + clf_alias[self.clf_str] + '.csv', index=False, header=False)
		# ====================================================================





		ax.legend(bbox_to_anchor=(1.32, 1), fontsize=12, loc='upper right', framealpha =0.6)
		if not self.show_full_xaxis:
			ax.set_xlim(0, max_x_lim * 1.5)
			if self.ylim:
				ax.set_ylim(0, self.ylim)



		remove_seed_genes_str = ''
		if len(genes_to_remove) > 0:
			remove_seed_genes_str = '.removed_' + str(len(genes_to_remove)) + '_seed_genes'
		xaxis_str = ''
		if self.show_full_xaxis:
		   xaxis_str = '.full_xaxis'

		fig.savefig(self.base_enrichment_dir + '/' + self.clf_str + xaxis_str + remove_seed_genes_str + '.pdf', bbox_inches='tight')
		plt.close()








def main():

	parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
	parser.add_argument("-c", dest="config_file", required=True,
			    help="Config file (.yaml) with run parameters [Required]\n\n")
	parser.add_argument("-o", dest="output_dir", help="Output directory name\n(absolute/relative path e.g. ./CKD, /tmp/Epilepsy-testing, etc.)\nIf it doesn't exist it will automatically be created [Required]\n\n", required=True)
	parser.add_argument("-e", dest="external_ranked_file", required=True,
			    help="Input file with external ranked gene list;\neither 1-column or 2-columns (with p-values in the 2nd column) [Required]\n\n")
	parser.add_argument("-t", dest="top_ratio", required=False, default=5,
			    help="Top percent ratio of mantis-ml predictions\nto overlap with the external ranked list (default: 5)\n\n")
	parser.add_argument("-m", dest="max_overlapping_genes", required=False, default=50,
			    help="Max. number of genes to retain that overlap\nmantis-ml and EXTERNAL_RANKED_FILE predictions (default: 50)\n\n")
	parser.add_argument("-y", dest="ylim", required=False,
			    help="Explicitly define y-axis max. limit (PHRED score value)\n\n")
	parser.add_argument("-f", "--full_xaxis", action="count", required=False,
			    help="Plot enrichment signal across the entire x-axis\nand not just for the significant part (or the MAX_OVERLAPPING_GENES)\nof the external ranked list\n\n")
	
	if len(sys.argv)==1:
		parser.print_help(sys.stderr)     
		sys.exit(1)

	args = parser.parse_args()


	config_file = args.config_file
	output_dir = args.output_dir
	external_ranked_file = args.external_ranked_file
	top_ratio = float(args.top_ratio) / 100
	max_overlapping_genes = int(args.max_overlapping_genes)
	ylim = None
	if args.ylim:
		ylim = float(args.ylim)
	show_full_xaxis = bool(args.full_xaxis)

	print("\nInput arguments:\n")
	print('- config_file:', config_file)
	print('- external_ranked_file:', external_ranked_file)
	print('- top_ratio:', str(100 * top_ratio) + '%')
	print('-max_overlapping_genes (if applicable):', max_overlapping_genes)
	print('-ylim:', ylim)
	print('- show_full_xaxis:', show_full_xaxis)
	print("\n")
	# ***************************


	cfg = Config(config_file, output_dir)


	# Read aggregated results from classifiers
	try:
		print("Reading all_clf.pkl ...")
		with open(str(cfg.superv_out / 'all_clf.pkl'), 'rb') as input:
			all_clf = pickle.load(input)
	except Exception as e:
		print(e)
		sys.exit("all_clf.pkl not found. Please run 'process_classifier_results.py' first.")



	# Read classifiers in descending order of avg. AUC
	sorted_classifiers = []
	avg_aucs_file = str(cfg.superv_out / 'Avg_AUC_per_classifier.txt')
	with open(avg_aucs_file) as fh:
		for line in fh:
			tmp_clf, tmp_auc = line.split('\t')
			sorted_classifiers.append(tmp_clf)

	print(sorted_classifiers)

	#classifiers = ['XGBoost']
	classifiers = sorted_classifiers[:]

	# Read seed genes
	seed_genes = all_clf[classifiers[0]].known_genes.tolist()

	genes_to_remove = [] 
	remove_seed_genes = False
	if remove_seed_genes:
		genes_to_remove = seed_genes


	pval_cutoff = 1 # Seto to 1, to include all


	# ------------------ (Nearly) Static options ------------------
	use_phred = True	# default: True
	collapsing_top_ratio = -1  # Set to -1 to use pval_cutoff instead



	for clf_str in classifiers:
	
		print('\n> Classifier: ' + clf_str)
		print('Overlapping with top ' + str(float(top_ratio) * 100) + '% of ' + clf_str + ' predictions ...')
		
		rank_overlap = ExternalRankingOverlap(cfg, clf_str, seed_genes, top_ratio=top_ratio, max_overlapping_genes=max_overlapping_genes, show_full_xaxis=show_full_xaxis, ylim=ylim)

		rank_overlap.read_external_ranked_gene_list(external_ranked_file)
		print(rank_overlap.external_ranked_df.head())
		print(rank_overlap.external_ranked_df.shape)

		rank_overlap.calc_stepwise_hypergeometric(all_clf, pval_cutoff=pval_cutoff,
							 collapsing_top_ratio=collapsing_top_ratio)


	# Get consensus of novel (and known) gene predictions
	for gene_class in ['Novel', 'Known']:
		cons_obj = Consensus_Gene_Predictions(config_file, output_dir, top_ratio, gene_class)
		cons_obj.run()



if __name__ == '__main__':
	main()
