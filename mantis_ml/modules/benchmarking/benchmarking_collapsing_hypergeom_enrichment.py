import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import hypergeom, mannwhitneyu
import pandas as pd
import numpy as np
import pickle
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 10)
import sys, os
import random
from pathlib import Path
from scipy.integrate import simps
from mantis_ml.config_class import Config


"""
    # ***** Benchmarking module *****
    - This module has been used to perform the benchmarking of mantis-ml and of other tools (Phenolyzer, ToppGene, ToppNet)
      against results from published rare-variant genetic association studies on Chronic Kidney Disease, Amyotrophic Lateral
      Sclerosis and Epilepsy.

    - The global variables 'run_external_benchmark' and 'benchmark_tool' need to be set to 'True' and to '[name_of_method'] 
      when benchmarking any of the external tools (Phenolyzer, ToppGene, ToppNet).
	
    - Benchmarking for mantis-ml can be performed by setting 'run_external_benchmark'to 'False'.
"""


clf_alias = {'ExtraTreesClassifier': 'ET', 'SVC': 'SVC', 'DNN': 'DNN', 'RandomForestClassifier': 'RF',
			 'XGBoost': 'XGB', 'GradientBoostingClassifier': 'GB', 'Stacking': 'Stacking'}

primary_analysis_types = ['primary', 'URV_AC_eq_1', 'v-AURORA-CUMC-all_dom_ultrarare_OO', 'Dom_LoF']


primary_analyses_dict = {'CKD': 'v-AURORA-CUMC-all_dom_ultrarare_OO', 'GGE': 'primary', 'ALS': 'Dom_LoF'}
synonymous_analyses_dict = {'CKD': 'v-AURORA-CUMC-all_dom_rare_syn', 'GGE': 'synonymous', 'ALS': 'Dom_not_benign'}

color_pallete = {'primary': '#33a02c', 'lof': '#fb9a99', 'common': '#a6cee3', 'synonymous': '#000000',
				 'Dom_LoF': '#33a02c', 'Dom_coding': '#fb9a99', 'Dom_not_benign': '#000000',
				 'Rec_LoF': '#feb24c', 'Rec_coding': '#a1d99b', 'Rec_not_benign': '#3182bd',
				 'URV_AC_eq_1': '#33a02c', 'URV_AC_lteq_3': '#000000',
				 'v-AURORA-CUMC-all_dom_ultrarare_OO': '#33a02c', 'v-AURORA-CUMC-all_dom_rare_LOF': '#33a02c',
				 'piv-mendelian-all_dom_rare_LOF': '#33a02c', 'i-all_AURORA_dom_rare_missense': '#ff7f00',
				 'viii-CUMC-all_dom_rare_mtr50': '#33a02c', 'v-AURORA-CUMC-all_dom_rare_missense_mtr50': '#000000',
				 'i-all_AURORA_dom_rare_syn': '#a6cee3', 'v-AURORA-CUMC-all_dom_rare_syn': '#3182bd',
				 'piv-mendelian-all_dom_rare_syn': '#756bb1', 'v-AURORA-CUMC-all-rec_syn': '#d95f0e', 'iii-CUMC-all_dom_rare_syn': '#c51b8a',
				 'shuffle': '#1f78b4'}

dataset_ylim = {'CKD': 60, 'GGE': 60, 'ALS': 60}

class CollapsingMantisMlOverlap:

	def __init__(self, cfg):
		self.cfg = cfg


	# ------------ Read collapsing results based on selected analysis type ------------
	def read_collapsing_ranking(self, analysis_type='primary', disease='GGE', pval_cutoff=0.05, collapsing_top_ratio=-1, genes_to_remove=[]):

		shuffle_file_path = str(self.cfg.out_root / ('../../' + input_dir + '/' + disease + '/shuffle_collapsing_ranking.' + disease + '.csv'))

		original_analysis_type = analysis_type   
		if analysis_type == 'shuffle':
			if os.path.exists(shuffle_file_path):
				collapsing_df = pd.read_csv(shuffle_file_path, index_col=0)
				return collapsing_df
			else:
				analysis_type = analysis_types[0]
				if disease == 'ALS':
					analysis_type = 'Dom_coding'
				print('Analsysis type to shuffle:', analysis_type)
				print('Oriinal:', original_analysis_type)

		collapsing_df = pd.read_csv(str(self.cfg.out_root / ('../../' + input_dir + '/' + disease + '/' + analysis_type + '_collapsing_ranking.' + disease + '.csv')),
									header=None, quotechar="'")

		collapsing_df.columns = ['Gene_Name', 'p-val']

		# Implement random sorting of rows with same value
		if disease in ['GGE', 'NAFE', 'ALS', 'IPF'] and analysis_type != 'primary':

			tmp_tuples = [(collapsing_df.loc[i, 'Gene_Name'], collapsing_df.loc[i, 'p-val']) for i in
						  range(len(collapsing_df))]

			sorted_tuples = sorted(tmp_tuples, key=lambda v: (v[1], random.random()))

			sorted_genes = [t[0] for t in sorted_tuples]
			sorted_pvals = [t[1] for t in sorted_tuples]
			sorted_df = pd.DataFrame({'Gene_Name': sorted_genes, 'p-val': sorted_pvals}, index=range(len(sorted_genes)))

			collapsing_df = sorted_df.copy()

		collapsing_df['collapsing_rank'] = collapsing_df.index.values + 1


		if collapsing_top_ratio != -1:
			collapsing_df = collapsing_df.loc[ 0:int(collapsing_top_ratio * collapsing_df.shape[0]), :]
		else:
			collapsing_df = collapsing_df.loc[collapsing_df['p-val'] <= pval_cutoff, :] 


		collapsing_df = collapsing_df.loc[ ~collapsing_df['Gene_Name'].isin(genes_to_remove), :]

		# Save shufled collapsing df to be used by all classifiers for consistency
		if original_analysis_type == 'shuffle':
			collapsing_df = self.shuffle_ranking(collapsing_df)
			collapsing_df.to_csv(shuffle_file_path)
   
		print(collapsing_df.head())


		return collapsing_df



	def shuffle_ranking(self, collapsing_df):
		collapsing_df = collapsing_df.sample(frac=1).reset_index(drop=True)
		collapsing_df['collapsing_rank'] = collapsing_df.index.values + 1

		return collapsing_df


	def calc_phred_score(self, pval):
		return -10 * np.log10(pval)


	def find_last_signif_gene_index(self, df):
		for i in range(df.shape[0]):
			if df.loc[i, 'p-val'] > collapsing_signif_thres:
				return i


	def calc_stepwise_hypergeometric(self, clf_str, all_clf, top_ratio, disease='GGE', pval_cutoff=0.05,
									 collapsing_top_ratio=-1, show_plots=False,
									 genes_to_remove=[]):

		fig, ax = plt.subplots(figsize=(18, 13))

		hypergeom_res_per_analysis_type = dict()
		signif_ratio_per_analysis_type = dict()

		signif_thres = 0.05
		signif_thres = self.calc_phred_score(signif_thres)
		ax.axhline(y=signif_thres, linestyle='--', color='red', label='p-val: 0.05')

		max_x_lim = -1

		# define filename identifiers based on selected method to benchmark
		method_identifier = clf_alias[clf_str]
		if run_external_benchmark:
			method_identifier = benchmark_tool


		for analysis_type in analysis_types:

			print('\n>>>> Analysis type:', analysis_type)
			print('\n-- Classifier:', clf_str)

			collapsing_df = self.read_collapsing_ranking(analysis_type=analysis_type, disease=disease, pval_cutoff=pval_cutoff, genes_to_remove=genes_to_remove)


			# *** Ad-hoc for bencmarking ***
			if run_external_benchmark:
				benchmark_input_dir = "../../../misc/overlap-collapsing-analyses/" + benchmark_tool + '/' + cfg.phenotype
				benchmark_intersection_genes_file = benchmark_input_dir + '/collapsing_genes_intersection.' + cfg.phenotype + '.txt'
	 
				benchmark_intersection_genes = []
				with open(benchmark_intersection_genes_file) as fh:
					for line in fh:
						line = line.rstrip()
						benchmark_intersection_genes.append(line)
				print(benchmark_intersection_genes[:10])
				print(len(benchmark_intersection_genes)) 


				# Make benchmarking more fair by looking into collapsing analysis genes that are also present in the Phenolyzer results
				print(collapsing_df.shape)
				collapsing_df = collapsing_df.loc[ collapsing_df.Gene_Name.isin(benchmark_intersection_genes), : ]
				print(collapsing_df.shape)


			M = collapsing_df.shape[0]
			print('Population Size:', M)


			hypergeom_pvals = []
			hypergeom_ordered_genes = []


			if not run_external_benchmark:
				clf = all_clf[clf_str]
				proba_df = clf.gene_proba_df
				proba_df = proba_df.iloc[:, ~proba_df.columns.isin(genes_to_remove)]
				print(proba_df.head())
				print(proba_df.shape)


				# Subset top 'top_ratio' % of mantis-ml predictions to overlap with collapsing results
				proba_df = proba_df.iloc[:, 0:int(top_ratio * proba_df.shape[1])]
				mantis_ml_top_genes = list(proba_df.columns.values)
				print(mantis_ml_top_genes[:10])
				print('mantis-ml top genes:', len(mantis_ml_top_genes))

			else:
				mantis_ml_top_genes = []
				with open(benchmark_input_dir + '/' + cfg.phenotype + '.' + benchmark_tool.lower() + '.ranked_genes.txt.collapsing_intersection') as fh:
					for line in fh:
						line = line.rstrip()
						gene, ranking = line.split(',')
						mantis_ml_top_genes.append(gene)			
				print(mantis_ml_top_genes[:10])
				# subset top_ratio % of external method rankings
				mantis_ml_top_genes = mantis_ml_top_genes[ : int(len(mantis_ml_top_genes) * top_ratio)]



			collapsing_df = collapsing_df.loc[collapsing_df['Gene_Name'].isin(mantis_ml_top_genes)]
			collapsing_df.reset_index(drop=True, inplace=True)


			n = collapsing_df.shape[0]
			print('Total number of Successes:', n)

			print(collapsing_df.head())

			# ************* Hypergeometric Test *************
			for x in range(collapsing_df.shape[0]):
				N = collapsing_df.iloc[x, 2]
				cur_pval = hypergeom.sf(x - 1, M, n, N)

				hypergeom_pvals = hypergeom_pvals + [cur_pval]

				cur_gene = collapsing_df.loc[x, 'Gene_Name']
				hypergeom_ordered_genes = hypergeom_ordered_genes + [cur_gene]
			# ***********************************************

			min_pval = min(hypergeom_pvals)
			hypergeom_pvals = [self.calc_phred_score(pval) for pval in hypergeom_pvals]

			hypergeom_res_per_analysis_type[analysis_type] = hypergeom_pvals


			# ------------------------- Start plotting ----------------------------
			linewidth = 1
			if analysis_type in primary_analysis_types:
				linewidth = 2
			ax.plot(hypergeom_pvals, color=color_pallete[analysis_type],
						label=analysis_type,
						linewidth=linewidth)


			last_signif_index = self.find_last_signif_gene_index(collapsing_df)
			if last_signif_index is None:
				last_signif_index = collapsing_df.shape[0]
			print('last_signif_index:', last_signif_index)
			if last_signif_index > max_x_lim:
				max_x_lim = last_signif_index


			ax.set_xlim(left=-0.5)
			ax.set_xlabel('Top ' + str(round(100 * top_ratio, 1)) + '% mantis-ml predicted genes', fontsize=14)
			#ax.axvline(x=last_signif_index, linestyle='--', linewidth=1, color=color_pallete[analysis_type], label=analysis_type + ' < ' + str(collapsing_signif_thres))
			ax.axvline(x=last_signif_index, linestyle='--', linewidth=1, color=color_pallete[analysis_type])


			y_label = 'Phred score from hypergeometric test\n(significance increasing in positive direction)'
			ax.set_ylabel(y_label, fontsize=14)

			print('Analysis type:', analysis_type, ' - Min. p-value:', str(min_pval))


			signif_ratio = round(100 * len([v for v in hypergeom_pvals if v > signif_thres]) / len(hypergeom_pvals), 2)
			signif_ratio_per_analysis_type[analysis_type] = analysis_type + ': ' + str(signif_ratio) + '%'


			if not run_external_benchmark:
				top_overlapping_genes = hypergeom_ordered_genes[:last_signif_index]
				mantis_ml_top_overlap_genes_ranking = [mantis_ml_top_genes.index(gene)+1 for gene in top_overlapping_genes]

				mantis_ml_top_genes_proba = clf.gene_proba_means.loc[top_overlapping_genes]


				if analysis_type in primary_analysis_types:

					merged_results_df = pd.DataFrame({'Gene_Name': top_overlapping_genes, 'mantis_ml_rank': mantis_ml_top_overlap_genes_ranking})
					merged_results_df = pd.merge(merged_results_df, clf.percentile_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
					merged_results_df = pd.merge(merged_results_df, collapsing_df, how='left', left_on='Gene_Name', right_on='Gene_Name')

					merged_results_df = pd.merge(merged_results_df, pd.DataFrame({'Gene_Name': seed_genes, 'Known_gene': 1}),
												 how='left', left_on='Gene_Name', right_on='Gene_Name')
					merged_results_df.fillna(0, inplace=True)

	
					merged_results_df.sort_values(by='mantis_ml_rank', ascending=True, inplace=True)
					merged_results_df.to_csv(Path(str(self.cfg.out_root / ('../../' + input_dir + '/Hypergeometric_results/mantis_ml_vs_collapsing.Top_' + str(top_ratio) + '.' + method_identifier + '.' + disease + '.csv'))), index=False)

					novel_overlaping_genes = merged_results_df.loc[ merged_results_df.Known_gene == 0, 'Gene_Name']
					novel_overlaping_genes.to_csv(Path(str(self.cfg.out_root / ('../../' + input_dir + '/Hypergeometric_results/Novel_overlaping_genes.Top_' + str(top_ratio) + '.' + method_identifier + '.' + disease + '.csv'))), index=False, header=False)

					known_overlaping_genes = merged_results_df.loc[ merged_results_df.Known_gene == 1, 'Gene_Name']
					known_overlaping_genes.to_csv(Path(str(self.cfg.out_root / ('../../' + input_dir + '/Hypergeometric_results/Known_overlaping_genes.Top_' + str(top_ratio) + '.' + method_identifier + '.' + disease + '.csv'))), index=False, header=False)





		mann_whitney_res_str = ''
		for i in range(len(analysis_types) - 1):
			for j in range(i + 1, len(analysis_types)):
				analysis_1 = analysis_types[i]
				analysis_2 = analysis_types[j]

				mw_pval = mannwhitneyu(hypergeom_res_per_analysis_type[analysis_1], hypergeom_res_per_analysis_type[analysis_2])
				pval_to_print = str(mw_pval.pvalue)
				mann_whitney_res_str += 'Mann-Whitney U [' + analysis_1 + '] vs [' + analysis_2 + ']: ' + pval_to_print + '\n'
		print(mann_whitney_res_str)

		# retrieve primary vs shuffle Mann Whitney U p-value
		for analysis_type in analysis_types:
			if analysis_type in primary_analysis_types:
				hypergeom_enrichment_pval =  mannwhitneyu(hypergeom_res_per_analysis_type[analysis_type], hypergeom_res_per_analysis_type['shuffle'])
				hypergeom_enrich_str = analysis_type + ' vs shuffle p-value:' + str(hypergeom_enrichment_pval.pvalue)
			
				ax.set_title('['+ clf_str + '] Hypergeometric test: Collapsing analysis vs mantis-ml predictions\n' + hypergeom_enrich_str, fontsize=20)
				break


		with open(Path(str(cfg.out_root) + '/../../' + input_dir + '/Hypergeometric_results/' + disease + '/' + method_identifier + '.top' + str(top_ratio)
						  + '_genes.Mann_Whitney_U.txt'), "w") as text_file:
			text_file.write(mann_whitney_res_str)


		ax.legend(bbox_to_anchor=(1.32, 1), fontsize=12, loc='upper right', framealpha =0.6)
		if not show_full_xaxis:

			ax.set_xlim(0, max_x_lim * 1.5)
			ax.set_ylim(0, dataset_ylim[disease])


		def calc_areas_under_plots(focus_on_signif_area=False, y_pos_offset=20):
			areas_str = '>> Entire area:\n\n'
			if focus_on_signif_area:
				areas_str = '>> Focusing on significant area only:\n\n' 
			areas_dict = {}

			for analysis_type in analysis_types:
				hypergeom_pvals = hypergeom_res_per_analysis_type[analysis_type]
				if focus_on_signif_area:
					hypergeom_pvals = [max(v-signif_thres, 0) for v in hypergeom_pvals]
					
				if not show_full_xaxis:
					hypergeom_pvals = hypergeom_pvals[ :max_x_lim]
  
				area = round(simps(hypergeom_pvals, list(range(len(hypergeom_pvals))) ), 2)
				areas_str += analysis_type + ' -- area: ' + str(area) + '\n'
				#print('len(hypergeom_pvals):', len(hypergeom_pvals))
				areas_dict[analysis_type] = area


			if areas_dict[synonymous_analyses_dict[disease]] != 0:
				primary_div_by_synonymous = round(areas_dict[primary_analyses_dict[disease]] / areas_dict[synonymous_analyses_dict[disease]], 2)
				areas_str += '> primary / synonymous ratio: ' + str(primary_div_by_synonymous) + '\n'
			else:
				areas_str += '> primary / synonymous ratio: NA\n'

			if areas_dict['shuffle'] != 0:
				primary_div_by_shuffle = round(areas_dict[primary_analyses_dict[disease]] / areas_dict['shuffle'], 2)
				areas_str += '> primary / shuffle ratio: ' + str(primary_div_by_shuffle) + '\n'
			else:
				areas_str += '> primary / shuffle ratio: NA\n'  

			print(areas_str)
			plt.text(1, dataset_ylim[disease] - y_pos_offset, areas_str)

		calc_areas_under_plots()
		calc_areas_under_plots(focus_on_signif_area=True, y_pos_offset=10)
		

			
		signif_ratio_str = ''
		for analysis_type in analysis_types:
			signif_ratio_str += '\n' + signif_ratio_per_analysis_type[analysis_type]


		if show_plots:
			plt.show()


		remove_seed_genes_str = ''
		if len(genes_to_remove) > 0:
			remove_seed_genes_str = '.removed_' + str(len(genes_to_remove)) + '_seed_genes'
		xaxis_str = ''
		if show_full_xaxis:
		   xaxis_str = '.full_xaxis'

		fig.savefig(str(self.cfg.out_root / ('../../' + input_dir + '/Hypergeometric_results/' + disease + '/' + method_identifier + '.' + disease + xaxis_str + remove_seed_genes_str + '.pdf')), bbox_inches='tight')



if __name__ == '__main__':

	config_file = sys.argv[1] # Path('../../config.yaml')
	run_external_benchmark = bool(int(sys.argv[2]))
	if run_external_benchmark:
		benchmark_tool = sys.argv[3]   # Phenolyzer, ToppGene, ToppNet

	# *** Optional parameters ***
	top_ratio = 0.05
	remove_seed_genes = 0
	show_full_xaxis = 0
	# ***************************

	
	cfg = Config(config_file)
	print(cfg.superv_out)
	print(type(cfg.superv_out))


	# ------- Ad-hoc to overlap with Generic classifier predictions -------
	#with open('../../../out/Generic-production/supervised-learning/all_clf.pkl', 'rb') as input:
	#	all_clf = pickle.load(input)
	#classifiers = ['ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'XGBoost', 'DNN']
	# ---------------------------------------------------------------------	 


	# Read aggregated results from classifiers
	if not run_external_benchmark:
		try:
			print("Reading all_clf.pkl")
			with open(str(cfg.superv_out / 'all_clf.pkl'), 'rb') as input:
				all_clf = pickle.load(input)
		except Exception as e:
			print(e)
			sys.exit("all_clf.pkl not found. Please run 'process_classifier_results.py' first.")
	else:
		all_clf = {}



	classifiers = list(all_clf.keys()) # ['ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'XGBoost', 'DNN', 'Stacking']
	print(classifiers)

	if run_external_benchmark:
		classifiers = ['XGBoost']


	if not run_external_benchmark:
		seed_genes = all_clf[classifiers[0]].known_genes.tolist()
	else:
		seed_genes = []



	# ============== Hypergeometric test parameters ==============
	# >>> For CKD - JASN 2019
	if 'CKD' in cfg.phenotype:
		analysis_types = ['v-AURORA-CUMC-all_dom_ultrarare_OO',
				  'i-all_AURORA_dom_rare_syn', 
				  'i-all_AURORA_dom_rare_missense', 
				  'v-AURORA-CUMC-all_dom_rare_syn',
				  'shuffle']
		input_dir = 'CKD_JASN_2019'
		disease = 'CKD'


	# # >>> For Epilepsy-LancetNeurology_2017
	if 'Epilepsy' in cfg.phenotype:
		print('Found Epilepsy!')
		analysis_types = ['primary', 'synonymous', 'common', 'shuffle']
		input_dir = 'Epilepsy-LancetNeurology_2017'
		disease = 'GGE' # GGE, NAFE

	# >>> For ALS
	if 'ALS' in cfg.phenotype:
		input_dir = 'ALS_Science_2015'
		analysis_types = ['Dom_LoF', 'Dom_coding', 'Dom_not_benign', 'Rec_coding', 'shuffle'] 
		disease = 'ALS'



	pval_cutoff = 1 # Seto to 1, to include all
	collapsing_signif_thres = 0.05 


	# ------------------ (Nearly) Static options ------------------
	use_phred = True	# default: True
	show_plots = False   # default: False
	collapsing_top_ratio = -1  # Set to -1 to use pval_cutoff instead


	base_input_dir = 'misc/overlap-collapsing-analyses'
	input_dir = base_input_dir + '/' + input_dir

	epilepsy_43_known_genes = pd.read_csv(str(cfg.out_root / ('../../' + base_input_dir + '/Epilepsy-LancetNeurology_2017/lancet_neurol_43_known_epilepsy_genes.txt')), header=None)
	epilepsy_43_known_genes = epilepsy_43_known_genes.iloc[:, 0].tolist()

	ckd_66_known_genes = pd.read_csv( str(cfg.out_root / ( '../../' + base_input_dir + '/CKD_JASN_2019/nejm_2018_66_known_CKD_genes.txt')), header=None)
	ckd_66_known_genes = ckd_66_known_genes.iloc[:, 0].tolist()


	# ***** Remove known genes for more rigorous validation *****
	genes_to_remove = [] # Default: use this assignment to keep all genes
	# genes_to_remove = seed_genes
	if remove_seed_genes:
	   if cfg.phenotype == 'CKD':
		   genes_to_remove = ckd_66_known_genes
	   elif cfg.phenotype == 'Epilepsy':	
		   genes_to_remove = epilepsy_43_known_genes
	print('Genes to remove:', len(genes_to_remove))
	# ----------------------------------------------------

	

	overlap_obj = CollapsingMantisMlOverlap(cfg)

	for clf_str in classifiers:
		print('Top ratio:', top_ratio, '%')
		overlap_obj.calc_stepwise_hypergeometric(clf_str, all_clf, top_ratio, disease=disease, pval_cutoff=pval_cutoff,
												 collapsing_top_ratio=collapsing_top_ratio, show_plots=show_plots,
												 genes_to_remove=genes_to_remove)

