import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import re
import os, sys
import random

from mantis_ml.modules.pre_processing.data_compilation.process_generic_features import ProcessGenericFeatures
from mantis_ml.config_class import Config



class ProcessFeaturesFilteredByDisease(ProcessGenericFeatures):

	def __init__(self, cfg):
		ProcessGenericFeatures.__init__(self, cfg)

	def process_hpo(self, include_terms, exclude_terms, annot_descr, conservative=True, save_to_file=False):
		'''
		Process Human Phenotype Ontology
		:param include_go_terms: list of strings that are queried as substrings of disease-associated terms 
		:param exclude_go_terms: list of strings to be excluded if they are substrings of GO terms
		:param annot_descr: disease/phenotype annotation string
		:param conservative: Boolean
		:return: hpo_df
		'''
		print("\n>> Compiling HPO features...")

		df = None

		if not conservative:
			# more inclusive
			df = pd.read_csv(self.cfg.data_dir / 'HPO/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt', sep='\t')
		else:
			# more conservative (default)
			df = pd.read_csv(self.cfg.data_dir / 'HPO/ALL_SOURCES_FREQUENT_FEATURES_genes_to_phenotype.txt', sep='\t')

		exclude_pattern = re.compile('|'.join(exclude_terms), re.IGNORECASE)
		if len(exclude_terms) > 0:
			df = df.loc[ ~df['HPO-Term-Name'].str.contains(exclude_pattern)]

		include_pattern = re.compile('|'.join(include_terms), re.IGNORECASE)
		seed_df = df.loc[ df['HPO-Term-Name'].str.contains(include_pattern)]

		hpo_selected_terms = seed_df['HPO-Term-Name']
		hpo_selected_terms = hpo_selected_terms.unique()


		known_genes_df = pd.DataFrame({'Gene_Name': seed_df['entrez-gene-symbol'].unique(), 'known_gene': 1})


		# TO-DO: test that hiding seed genes works for the non-Generic classifier too
		if self.cfg.hide_seed_genes_ratio > 0:
			sample_known_genes_df = known_genes_df.sample(frac=(1-self.cfg.hide_seed_genes_ratio))

			hidden_seed_genes = pd.Series(list(set(known_genes_df.Gene_Name) - set(sample_known_genes_df.Gene_Name)))
			hidden_seed_genes.to_csv(str(self.cfg.out_data_dir / 'hidden_seed_genes.txt'), index=None, header=False)

			known_genes_df = sample_known_genes_df

		if save_to_file:
			known_genes_df.to_csv(self.cfg.data_dir / ("HPO/compiled_known_" + annot_descr + "_genes.tsv"), sep='\t', index=None)

		print("Total HPO Genes associated with selected pattern: {0}".format(known_genes_df.shape[0]))

		return known_genes_df, hpo_selected_terms


	def process_omim(self, pattern):
		df = pd.read_csv(self.cfg.data_dir / ('omim/' + pattern + '_genes.txt'), sep='\t', header=None)
		df.columns = ['Gene_Name']
		df[self.cfg.Y] = 1

		if self.cfg.hide_seed_genes_ratio > 0:
			sample_df = df.sample(frac=(1-self.cfg.hide_seed_genes_ratio))

			hidden_seed_genes = pd.Series(list(set(df.Gene_Name) - set(sample_df.Gene_Name)))
			hidden_seed_genes.to_csv(str(self.cfg.out_data_dir / 'hidden_seed_genes.txt'), index=None, header=False)

			df = sample_df

		return df


	def process_gtex_features(self, save_to_file=False):
		'''
		Get Protein TPM expression and Rank from GTEx
		:param pattern: 
		:return: 
		'''
		print("\n>> Compiling GTEx features...")
		all_patterns = self.cfg.seed_include_terms[:]
		all_patterns.extend(self.cfg.additional_include_terms)
		print('All patterns:', all_patterns)
		
		exclude_pattern = re.compile('|'.join(self.cfg.exclude_terms), re.IGNORECASE)


		full_df = pd.read_csv(self.cfg.data_dir / 'gtex/RNASeq/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct', sep='\t')
		#print(full_df.head())
		all_tissue_cols = full_df.columns.values
		print(all_tissue_cols)
		all_selected_tissue_cols = []

		return_gtex_df = pd.DataFrame()
		if not self.cfg.generic_classifier:
			for pattern in all_patterns:
				selected_tissue_cols = [c for c in full_df.columns if re.compile(pattern, re.IGNORECASE).search(c)]
	
				if len(self.cfg.exclude_terms) > 0:
					selected_tissue_cols = [c for c in selected_tissue_cols if not re.compile(exclude_pattern).search(c)]
					
				if len(selected_tissue_cols) == 0: # return if no matching columns exist
					continue
				else:
					print('\n', pattern, ':', selected_tissue_cols)


				#print('Tissue/Disease pattern:', selected_tissue_cols)
				all_selected_tissue_cols.extend(selected_tissue_cols)

				df = full_df[['Gene_Name', 'gene_id'] + selected_tissue_cols]
				#print(df.columns)
				#print(df.shape)

				agg_df = df.groupby('Gene_Name').agg('sum')
				#print(agg_df.columns)
				#print(agg_df.shape)


				pattern = pattern.replace(' ', '')
				total_tissue_expr = 'GTEx_' + pattern + '_TPM_expression'
				sum_gtex_df = pd.DataFrame(agg_df.sum(axis=1), columns=[total_tissue_expr])

				# limit to default HGNC gene-set
				sum_gtex_df = sum_gtex_df.reindex(self.cfg.hgnc_genes_series)
				sum_gtex_df.fillna(0, inplace=True)
				sum_gtex_df.sort_values(by=total_tissue_expr, inplace=True, ascending=False)

				sum_gtex_df.reset_index(inplace=True)
				sum_gtex_df.columns.values[0] = 'Gene_Name'

				# Assign Rank = len(default_gene_set) to all genes with total expression less than the median among all genes
				tissue_rank = 'GTEx_' + pattern + '_Expression_Rank'
				sum_gtex_df[tissue_rank] = sum_gtex_df.index + 1
				sum_gtex_df.loc[sum_gtex_df[total_tissue_expr] < int(sum_gtex_df[total_tissue_expr].median()), tissue_rank] = len(self.cfg.hgnc_genes_series)
				#print(sum_gtex_df.head())


				if len(return_gtex_df) > 0:
					return_gtex_df = pd.merge(return_gtex_df, sum_gtex_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
				else:
					return_gtex_df = sum_gtex_df

			if save_to_file:
				return_gtex_df.to_csv(self.cfg.data_dir / ('gtex/RNASeq/'+ self.cfg.phenotype + '_GTEx_expression_features.tsv'), sep='\t', index=None)
			# TODO: keep expression in each tissue as a separate feature - do not sum


		else:

			full_df.drop(['gene_id'], axis=1, inplace=True)

			full_df = full_df.rename(columns={col: col.replace(' ', '_').replace('-', '') for col in full_df.columns if col != self.cfg.gene_name})
			full_df = full_df.rename(columns={col: col.replace('(', '').replace(')', '').replace('__', '_') for col in full_df.columns if col != self.cfg.gene_name})
			full_df = full_df.rename(columns={col: 'GTEx_' + col + '_TPM_expression' for col in full_df.columns if col != self.cfg.gene_name})

			return_gtex_df = full_df.groupby('Gene_Name').agg('sum')
			return_gtex_df.reset_index(inplace=True)

			print(return_gtex_df.head())
			# TODO: keep expression in each tissue as a separate feature - do not sum


		print('GTEx:', return_gtex_df.head())

		return return_gtex_df, all_selected_tissue_cols, all_tissue_cols




	def process_protein_atlas_features(self, include_terms, exclude_terms, verbose=False, save_to_file=False):
		'''
		Get protein expression levels and RNA TPM for Human Protein Atlas
		:param pattern: 
		:return: 
		'''

		#TODO: Currently collapsing values from multiple tissues -- see if this can be untangled in future version

		print("\n>> Compiling Human Protein Atlas features...")

		tissue_str = self.cfg.phenotype.lower()

		# pattern = re.compile('|'.join(include_terms), re.IGNORECASE)
		# pattern = re.compile(pattern, re.IGNORECASE)

		include_pattern = re.compile('|'.join(include_terms), re.IGNORECASE)
		exclude_pattern = re.compile('|'.join(exclude_terms), re.IGNORECASE)


		# normal_tissue.tsv.gz
		normal_df = pd.read_csv(self.cfg.data_dir / 'human_protein_atlas/normal_tissue.tsv.gz', sep='\t')
		print(normal_df.shape)

		all_normal_tissues = normal_df['Tissue'].unique().tolist()
		if verbose:
			print(all_normal_tissues)

		if not self.cfg.generic_classifier:
			print('[normal] Keeping only entries from tissue: {0} ...'.format(include_pattern))
			# normal_df = normal_df.loc[ normal_df.Tissue.str.contains(pattern)]
			if len(exclude_terms) > 0:
				normal_df = normal_df.loc[~normal_df['Tissue'].str.contains(exclude_pattern)]
			normal_df = normal_df.loc[normal_df['Tissue'].str.contains(include_pattern)]



		print("[normal] Removing entries with Reliability = 'Uncertain...'")
		normal_df = normal_df.loc[ normal_df.Reliability != 'Uncertain']
		print(normal_df.shape)
		selected_normal_tissues = normal_df['Tissue'].unique().tolist()

		normal_df = normal_df.iloc[:, [1,3,4]]
		print(normal_df.head())

		def generic_collapse_normal_tissue_expression(df, target_col):
			"""
			# Category coding for aggregation across multiple cell lines in same tissue
			- Not detected: 0
			- Low: 0
			- Medium: 1
			- High: 1
	
			Rule: Keeping the max. value after aggregation by same Gene Name
			"""
			print("Collapsing normal tissue expression from entries with same gene (keeping highest)...")
			df.replace({target_col: {'Not detected': 0, 'Low': 0, 'Medium': 1, 'High': 1}}, inplace=True)

			df[target_col] = df[target_col].astype(str)

			tmp_df = pd.DataFrame(df.groupby(['Gene name', 'Cell type'])[target_col].agg('|'.join))

			# tmp_df['delim_cnt'] = tmp_df[target_col].apply(lambda x: x.count('|'))
			tmp_df['final_level'] = tmp_df[target_col].apply(lambda x: max(x.split('|')))
			tmp_df['final_level'] = tmp_df['final_level'].astype(int)
			tmp_df.drop([target_col], axis=1, inplace=True)
			print(tmp_df.head())

			# tmp_df = tmp_df.unstack(fill_value=0)
			tmp_df.reset_index(inplace=True)
			tmp_df = tmp_df.pivot(index='Gene name', columns='Cell type', values='final_level')
			tmp_df.fillna(0, inplace=True)
			print(tmp_df.head())
			print(tmp_df.shape)

			tmp_df = tmp_df.rename(columns={col: col.replace(' ', '_') for col in tmp_df.columns})
			tmp_df = tmp_df.rename(columns={col: 'ProteinAtlas_' + col + '_Expr_Flag' for col in tmp_df.columns})
			tmp_df.index.names = ['Gene_Name']
			tmp_df.reset_index(inplace=True)


			tissue_str = 'generic'
			if save_to_file:
				tmp_df.to_csv(self.cfg.data_dir / ('human_protein_atlas/human_protein_atlas_' + tissue_str + '_expression_levels.tsv'), sep='\t', index=None)

			return tmp_df


		def collapse_normal_tissue_expression(df, target_col):
			"""
			# Category coding for aggregation across multiple cell lines in same tissue
			- Not detected: 0
			- Low: 1
			- Medium: 3
			- High: 8
	
			Rule: Keeping the max. value after aggregation by same Gene Name
			"""
			print("Collapsing normal tissue expression from entries with same gene (keeping highest)...")
			df.replace({target_col: {'Not detected': 0, 'Low': 1, 'Medium': 3, 'High': 8}}, inplace=True)

			df[target_col] = df[target_col].astype(str)

			tmp_df = pd.DataFrame(df.groupby('Gene name')[target_col].agg('|'.join))

			tmp_df['delim_cnt'] = tmp_df[target_col].apply(lambda x: x.count('|'))
			tmp_df['final_level'] = tmp_df[target_col].apply(lambda x: max(x.split('|')))
			print(tmp_df.head())
			tmp_df['Gene name'] = tmp_df.index.copy()

			final_df = tmp_df[['Gene name', 'final_level']].copy()
			final_df.rename(columns={'final_level': target_col}, inplace=True)

			final_df.replace({target_col: {'0': 'Not_detected', '1': 'Low', '3': 'Medium', '8': 'High'}}, inplace=True)
			final_df.columns = ['Gene_Name', 'ProteinAtlas_gene_expr_levels']
			# print(final_df.head())

			if save_to_file:
				final_df.to_csv(self.cfg.data_dir / ('human_protein_atlas/human_protein_atlas_' + tissue_str + '_expression_levels.tsv'), sep='\t', index=None)
			return final_df

		target_col = 'Level'
		expr_levels_df = None
		if not self.cfg.generic_classifier:
			expr_levels_df = collapse_normal_tissue_expression(normal_df, target_col)
		else:
			expr_levels_df = generic_collapse_normal_tissue_expression(normal_df, target_col)
		print(expr_levels_df.head())
		print(expr_levels_df.shape)



		# =========== rna_tissue.tsv.gz ============
		rna_df = pd.read_csv(self.cfg.data_dir / 'human_protein_atlas/rna_tissue.tsv.gz', sep='\t')
		# print(rna_df.head())
		print(rna_df.shape)

		all_rna_samples = rna_df['Sample'].unique().tolist()

		if not self.cfg.generic_classifier:
			print('[rna] Keeping only entries from tissue: {0} ...'.format(include_pattern))
			# rna_df = rna_df.loc[rna_df.Sample.str.contains(pattern)]
			if len(exclude_terms) > 0:
				rna_df = rna_df.loc[~rna_df['Sample'].str.contains(exclude_pattern)]
			rna_df = rna_df.loc[rna_df['Sample'].str.contains(include_pattern)]

		selected_rna_samples = rna_df['Sample'].unique().tolist()

		def generic_collapse_rna_expression(df, target_col):
			"""
			# Aggregate TPM values across multiple entries of same gene
			"""

			print("Collapsing rna tissue expression from entries with same gene (sum)...")
			print(df.shape)
			tmp_df = df.groupby(['Gene name', 'Sample']).sum()
			print(tmp_df.head())

			tmp_df.reset_index(inplace=True)
			tmp_df = tmp_df.pivot(index='Gene name', columns='Sample', values='Value')
			tmp_df.fillna(0, inplace=True)
			print(tmp_df.head())
			print(tmp_df.shape)

			tmp_df = tmp_df.rename(columns={col: col.replace(' ', '_') for col in tmp_df.columns})
			tmp_df = tmp_df.rename(columns={col: 'ProteinAtlas_' + col + '_RNA_Expr_TPM' for col in tmp_df.columns})
			tmp_df.index.names = ['Gene_Name']
			tmp_df.reset_index(inplace=True)

			if save_to_file:
				tissue_str = 'generic'
				tmp_df.to_csv(self.cfg.data_dir / ('human_protein_atlas/human_protein_atlas_' + tissue_str + '_rna_expression_tpm.tsv'),
							  sep='\t', index=None)

			return tmp_df

		def collapse_rna_expression(df, target_col):
			"""
			# Aggregate TPM values across multiple entries of same gene
			"""

			print("Collapsing rna tissue expression from entries with same gene (sum)...")
			print(df.shape)
			df = df.groupby('Gene name').sum()

			df['Gene name'] = df.index.copy()
			df = df[['Gene name', target_col]]
			df.columns = ['Gene_Name', 'ProteinAtlas_RNA_expression_TMP']

			if save_to_file:
				df.to_csv(self.cfg.data_dir / ('human_protein_atlas/human_protein_atlas_' + tissue_str + '_rna_expression_tpm.tsv'), sep='\t', index=None)

			return df

		target_col = 'Value'
		expr_tpm_df = None
		if not self.cfg.generic_classifier:
			expr_tpm_df = collapse_rna_expression(rna_df, target_col)
		else:
			expr_tpm_df = generic_collapse_rna_expression(rna_df, target_col)
		print(expr_tpm_df.shape)
		print(expr_tpm_df.head())

		# prot_atlas_df = expr_tpm_df # To keep only RNA expression TPM values
		prot_atlas_df = pd.merge(expr_levels_df, expr_tpm_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')




		return prot_atlas_df, selected_normal_tissues, all_normal_tissues, selected_rna_samples, all_rna_samples




	def process_msigdb_go_features(self, include_go_terms, exclude_go_terms, min_go_set_length=150, verbose=False, save_to_file=False):
		'''
		Get GO information from msigdb for genes associated with particular terms
		
		:param include_go_terms: list of strings that are queried as substrings of GO terms 
		:param exclude_go_terms: list of strings to be excluded if they are substrings of GO terms
		:return: 
		'''
		print("\n>> Compiling MsigDB GO features...")

		generic_go_file = self.cfg.data_dir / ('msigdb/tables_per_gene_set/generic.min_go_set_len' + str(min_go_set_length) + '_GO_Features.tsv')
		if os.path.exists(generic_go_file) and self.cfg.generic_classifier:
			msigdb_go_df = pd.read_csv(generic_go_file, sep='\t')
			return msigdb_go_df

		full_go_file = (self.cfg.data_dir / 'msigdb/tables_per_gene_set/c5.all.v6.2.symbols.gmt')

		gene_lists_per_go_term = dict()
		if not self.cfg.generic_classifier:
			for t in include_go_terms:
				gene_lists_per_go_term[t] = []


		selected_go_terms = []
		cnt = 0
		with open(full_go_file) as fh:
			for line in fh:
				line = line.rstrip()
				vals = line.split('\t')
				del vals[1]

				cur_go_field = vals[0]
				if len(exclude_go_terms) > 0:
					if any(re.compile(excl_term, re.IGNORECASE).search(cur_go_field) for excl_term in exclude_go_terms):
						continue

				for t in range(len(include_go_terms)):
					incl_term = include_go_terms[t]
					if (re.compile(incl_term, re.IGNORECASE).search(cur_go_field)):
						selected_go_terms.append(cur_go_field)
						genes = vals[1:]
						if not self.cfg.generic_classifier:
							gene_lists_per_go_term[incl_term].extend(genes)
						else:
							gene_lists_per_go_term[cur_go_field] = genes

						if verbose:
							print(cur_go_field)

		new_gene_lists_per_go_term = dict()
		for term in gene_lists_per_go_term.keys():
			new_gene_lists_per_go_term[term] = list(set(gene_lists_per_go_term[term]))

			if  self.cfg.generic_classifier and (len(new_gene_lists_per_go_term[term]) < min_go_set_length):
				del new_gene_lists_per_go_term[term]

		gene_lists_per_go_term = new_gene_lists_per_go_term.copy()

		msigdb_go_df = pd.DataFrame()
		for term in gene_lists_per_go_term.keys():
			tmp_df = pd.DataFrame({'Gene_Name': gene_lists_per_go_term[term], term: 1})
			tmp_df = tmp_df[['Gene_Name', term]]
			print(tmp_df.shape)

			if msigdb_go_df.shape[0] > 0:
				msigdb_go_df = pd.merge(msigdb_go_df, tmp_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
			else:
				msigdb_go_df = tmp_df

		msigdb_go_df.fillna(0, inplace=True)
		# print(msigdb_go_df.loc[msigdb_go_df.Gene_Name.isin(['PKD1', 'PKD2', 'NOTCH1'])])


		go_cols = [(x.replace(' ', '_')) for x in msigdb_go_df.columns if x != 'Gene_Name']
		if not self.cfg.generic_classifier:
			go_cols = [('GO_' + x) for x in msigdb_go_df.columns if x != 'Gene_Name']


		go_cols = ['Gene_Name'] + go_cols
		print(go_cols)
		msigdb_go_df.columns = go_cols

		file_prefix = self.cfg.phenotype
		if self.cfg.generic_classifier:
			file_prefix = 'generic.min_go_set_len' + str(min_go_set_length)

		if save_to_file:
			msigdb_go_df.to_csv(self.cfg.data_dir / ('msigdb/tables_per_gene_set/' + file_prefix + '_GO_Features.tsv'), sep='\t', index=None)


		return msigdb_go_df, selected_go_terms



	def process_mgi_features(self, include_terms, exclude_terms, verbose=False, save_to_file=False):
		'''
		Retrieve human genes with mouse orthologs that have a particular phenotype
		:param include_terms: terms to look for in the mammalian phenotype
		:param exclude_terms: terms to exclude from mammalian phenotype search
		:return: 
		'''
		print(">> Compiling MGI phenotype-associated genes features...")

		include_terms.extend(self.cfg.additional_include_terms)

		query_human_pheno_df = pd.read_csv(self.cfg.data_dir / 'mgi/hmd_human_pheno.processed.rpt', sep='\t')
		query_human_pheno_df.fillna('', inplace=True)

		# exclude query terms
		exclude_pattern = re.compile('|'.join(exclude_terms), re.IGNORECASE)
		if len(exclude_terms) > 0:
			query_human_pheno_df = query_human_pheno_df.loc[~query_human_pheno_df['human_phenotypes'].str.contains(exclude_pattern)]

		# include query terms
		include_pattern = re.compile('|'.join(include_terms), re.IGNORECASE)
		query_human_pheno_df = query_human_pheno_df.loc[
		query_human_pheno_df['human_phenotypes'].str.contains(include_pattern)]

		selected_mgi_phenotypes = query_human_pheno_df['human_phenotypes'].unique().tolist()
		if verbose:
			print(selected_mgi_phenotypes)

		mgi_pheno_genes = query_human_pheno_df['Human Marker Symbol'].unique()
		mgi_pheno_df = pd.DataFrame({'Gene_Name': mgi_pheno_genes, 'MGI_mouse_knockout_feature': 1})

		return  mgi_pheno_df, selected_mgi_phenotypes



	def process_inweb(self, seed_genes):
		print("\n>> Compiling InWeb features...")

		def get_seed_genes_overlap(interacting_genes):
			interacting_genes = eval(interacting_genes)
			overlapping_genes = list(set(list(interacting_genes)) & set(seed_genes))
			perc_overlap = len(overlapping_genes) / len(interacting_genes)

			return perc_overlap


		inferred_df = pd.read_csv(self.cfg.data_dir / 'in_web/inferred_pairwise_interactions.tsv', sep='\t', index_col=None)
		inferred_df['Inferred_seed_genes_overlap'] = inferred_df['interacting_genes'].apply(get_seed_genes_overlap)
		inferred_df.drop(['interacting_genes'], axis=1, inplace=True)
		print(inferred_df.head())

		experim_df = pd.read_csv(self.cfg.data_dir / 'in_web/experimental_pairwise_interactions.tsv', sep='\t', index_col=None)
		experim_df['Experimental_seed_genes_overlap'] = experim_df['interacting_genes'].apply(get_seed_genes_overlap)
		experim_df.drop(['interacting_genes'], axis=1, inplace=True)
		print(experim_df.head())

		inweb_df = pd.merge(experim_df, inferred_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
		inweb_df.fillna(0, inplace=True)

		return inweb_df



	def run_all(self):

		seed_include_terms = self.cfg.seed_include_terms
		exclude_terms = self.cfg.exclude_terms

		# Get Seed genes (positive data points)
		if not self.cfg.generic_classifier:
			# HPO
			seed_genes_df, _ = self.process_hpo(seed_include_terms, exclude_terms, self.cfg.phenotype)
			print('HPO:', seed_genes_df.shape)
		else:
			seed_genes_df = self.process_omim(self.cfg.generic_classifier)

		print(seed_genes_df.shape)
		print(seed_genes_df.head())
		if seed_genes_df.shape[0] == 0:
			sys.exit('[Error] No seed genes found for current terms:' + ','.join(seed_include_terms))


		# GTEx
		gtex_df, _, _ = self.process_gtex_features()
		print('GTEx:', gtex_df.shape)


		# GWAS
		if not self.cfg.generic_classifier:
			gwas_include_terms = seed_include_terms[:]
			gwas_include_terms.extend(self.cfg.additional_include_terms)
			pattern_lists = [gwas_include_terms, exclude_terms]
			print(pattern_lists)

			tissue_gwas_df = self.process_gwas_features(pattern_lists=pattern_lists, search_term=self.cfg.phenotype)
			print('GWAS:', tissue_gwas_df.shape)


		# Protein Atlas
		protatlas_include_terms = self.cfg.seed_include_terms[:]
		protatlas_include_terms.extend(self.cfg.additional_include_terms)

		prot_atlas_df, _, _, _, _ = self.process_protein_atlas_features(protatlas_include_terms, self.cfg.exclude_terms)
		print('Human Protein Atlas:', prot_atlas_df.shape)
		print(prot_atlas_df.shape)


		# MSigDB
		msigdb_include_terms = self.cfg.seed_include_terms[:]
		msigdb_include_terms.extend(self.cfg.additional_include_terms)

		if self.cfg.generic_classifier:
			msigdb_include_terms = ['.*']
			exclude_terms = []
		msigdb_go_df, _ = self.process_msigdb_go_features(msigdb_include_terms, exclude_terms)
		print('MSigDB:', msigdb_go_df.shape)
		print(msigdb_go_df.iloc[:, 0:10].head())
		print(msigdb_go_df.columns.values[:10])


		# MGI
		if not self.cfg.generic_classifier:
			mgi_include_terms = seed_include_terms[:]
			mgi_include_terms.extend(self.cfg.additional_include_terms)

			mgi_pheno_df, _ = self.process_mgi_features(mgi_include_terms, exclude_terms)
			print('MGI:', mgi_pheno_df.shape)


		# InWeb_IM
		seed_genes = seed_genes_df['Gene_Name'].tolist()
		inweb_df = self.process_inweb(seed_genes)
		print('InWeb_IM:', inweb_df.shape)



		print("\n>> Merging all data frames together...")
		filtered_by_disease_features_df = pd.merge(inweb_df, seed_genes_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
		print(filtered_by_disease_features_df.shape)
		if gtex_df.shape[0] > 0:
			filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, gtex_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
			print(filtered_by_disease_features_df.shape)
		if prot_atlas_df.shape[0] > 0:
			filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, prot_atlas_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
			print(filtered_by_disease_features_df.shape)

		if msigdb_go_df.shape[0] > 0:
			filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, msigdb_go_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
			print(filtered_by_disease_features_df.shape)


		if not self.cfg.generic_classifier:
			if tissue_gwas_df.shape[0] > 0:
				filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, tissue_gwas_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
				print(filtered_by_disease_features_df.shape)

			if mgi_pheno_df.shape[0] > 0:
				filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, mgi_pheno_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
				print(filtered_by_disease_features_df.shape)


		# ---------------------------------------------------------
		# Impute 'known_gene', 'GO_*', 'MGI_mouse_knockout_feature' & 'ProteinAtlas_gene_expr_levels' with zero, for all genes that don't have a '1' value:
		# these values are not missing data but rather represent a 'False'/zero feature value.
		filtered_by_disease_features_df['known_gene'].fillna(0,inplace=True)

		protatlas_cols = [c for c in prot_atlas_df.columns.values if c != 'Gene_Name']
		for c in protatlas_cols:
			if c in filtered_by_disease_features_df.columns:
				filtered_by_disease_features_df[c].fillna(0, inplace=True)

		go_cols = [c for c in msigdb_go_df.columns.values if c != 'Gene_Name']
		for c in go_cols:
			if c in filtered_by_disease_features_df.columns:
				filtered_by_disease_features_df[c].fillna(0,inplace=True)

		if 'MGI_mouse_knockout_feature' in filtered_by_disease_features_df.columns:
			filtered_by_disease_features_df['MGI_mouse_knockout_feature'].fillna(0,inplace=True)
		# ---------------------------------------------------------


		if self.cfg.random_seeds:
			print(filtered_by_disease_features_df.loc[ filtered_by_disease_features_df.known_gene == 1, :].shape)

			total_seed_genes = filtered_by_disease_features_df.loc[ filtered_by_disease_features_df.known_gene == 1, :].shape[0]
			# reset known genes labels with '0' value for all genes
			filtered_by_disease_features_df.loc[:, 'known_gene'] = 0
			print(filtered_by_disease_features_df.loc[ filtered_by_disease_features_df.known_gene == 1, :].shape)

			# select random indexes
			random_seed_indexes = np.random.choice(list(range(filtered_by_disease_features_df.shape[0])), size=total_seed_genes, replace=False).tolist()
			print(len(random_seed_indexes))

			# assign '1' value for 'known_gene' label to random genes indicated by the generated random indexes
			filtered_by_disease_features_df.loc[ random_seed_indexes, 'known_gene'] = 1
			print(filtered_by_disease_features_df.loc[ filtered_by_disease_features_df.known_gene == 1, :].shape)



		filtered_by_disease_features_df.to_csv(self.cfg.filtered_by_disease_feature_table, sep='\t', index=None)
		print("Saved to {0}".format(self.cfg.filtered_by_disease_feature_table))
		print(filtered_by_disease_features_df.shape)

		duplicate_gene_names = filtered_by_disease_features_df.Gene_Name[filtered_by_disease_features_df.Gene_Name.duplicated()].unique()
		if len(duplicate_gene_names) > 0:
			print('[Error] Duplicate Gene Names:')
			print(duplicate_gene_names)
			sys.exit(-1)


if __name__ == '__main__':

	config_file = sys.argv[1] #'../../../config.yaml'
	cfg = Config(config_file)

	proc = ProcessFeaturesFilteredByDisease(cfg)
	#gtex_df, all_selected_tissue_cols, _ = proc.process_gtex_features()
	proc.run_all()

