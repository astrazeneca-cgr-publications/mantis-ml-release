# -*- coding: utf-8 -*-
from collections import Counter
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import sys, os
import re

from mantis_ml.modules.pre_processing.data_compilation.process_features_filtered_by_disease import ProcessFeaturesFilteredByDisease
from mantis_ml.config_class import Config



class MantisMlProfiler:

	def __init__(self, config_file, output_dir, verbose=False):

		self.config_file = config_file
		self.output_dir = output_dir
		self.verbose = verbose

		# common strings to exclude from profiling
		self.eng_stopwords = self.get_english_stopwords()
		self.custom_bullet = u'\u2022' * 5
		self.line_spacer = '\n' * 6
	

	# Disable print to stdout
	def blockPrint(self):
		sys.stdout = open(os.devnull, 'w')

	# Restore print to stdout
	def enablePrint(self):
		sys.stdout = sys.__stdout__

	def bordered(self, text):

		lines = text.splitlines()
		width = max(len(s) for s in lines)

		res = ['┌' + '─' * width + '┐']
		for s in lines:
			res.append('│' + (s + ' ' * width)[:width] + '│')
			res.append('└' + '─' * width + '┘')	 
		
		return '\n'.join(res)



	def assess_hpo_filtered_output(self, proc_obj, cfg):
		print(self.line_spacer + "-----------------   Assessing HPO filtering [config parameters: 'seed_include_terms']	-----------------\n")

		print("- Provided 'seed_include_terms':")
		print(cfg.seed_include_terms)

		seed_df, hpo_selected_terms = proc_obj.process_hpo(cfg.seed_include_terms, cfg.exclude_terms, cfg.phenotype)

		selected_genes = seed_df['Gene_Name'].tolist()
		if self.verbose:
			print('\n' + self.bordered(self.custom_bullet + ' Selected HPO genes ' + self.custom_bullet))
			print(selected_genes)

		print('\n' + self.bordered(self.custom_bullet + ' Selected HPO disease-associated terms ' + self.custom_bullet))
		print(sorted(list(hpo_selected_terms)))

		hpo_selected_terms_expanded = [s.split() for s in hpo_selected_terms]
		hpo_selected_terms_expanded = [item.lower() for sublist in hpo_selected_terms_expanded for item in sublist]

		# remove stopwords
		hpo_selected_terms_expanded = [w for w in hpo_selected_terms_expanded if w not in self.eng_stopwords]

		# remove digits
		hpo_selected_terms_expanded = [w for w in hpo_selected_terms_expanded if not w.isdigit()]

		if self.verbose:
			count_hpo_terms = Counter(hpo_selected_terms_expanded)
			print('\n' + u'\u2022' + ' Most common strings in filtered HPO phenotype terms:')
			for s, count in count_hpo_terms.most_common():
				if count == 1:
					continue
				print(s + ':', count)



	def assess_gtex_filtered_output(self, proc_obj, cfg):

		print(self.line_spacer + "-----------------	Assessing GTEx filtering [config parameters: 'tissue' and 'additional_tissues']	-----------------\n")
		print("\n- Provided 'seed_include_terms':")
		print(cfg.seed_include_terms)
		print("\n- Provided 'additional_include_terms':")
		print(cfg.additional_include_terms)
		#print("- Provided 'tissue':")
		#print(cfg.tissue)
		#print("\n- Provided 'additional_tissues':")
		#print(cfg.additional_tissues)

		self.blockPrint()
		_, selected_tissue_cols, all_tissue_cols = proc_obj.process_gtex_features()
		self.enablePrint()

		all_tissue_cols = list(all_tissue_cols)
		selected_tissue_cols = list(selected_tissue_cols)
		
		if self.verbose:
			print('\nAvailable GTEx tissues:')
			print(sorted(all_tissue_cols))

		print('\n' + self.bordered(self.custom_bullet + ' Selected GTEx tissues ' + self.custom_bullet))
		print(sorted(selected_tissue_cols))



	def assess_proteinatlas_filtered_output(self, proc_obj, cfg):

		print(self.line_spacer + "-----------------	Assessing Protein Atlas filtering [config parameters: 'tissue', 'seed_include_terms', 'additional_include_terms']	-----------------\n")
		print("- Provided 'tissue':")
		#print(cfg.tissue)
		print("\n- Provided 'seed_include_terms':")
		print(cfg.seed_include_terms)
		print("\n- Provided 'additional_include_terms':")
		print(cfg.additional_include_terms)

		#protatlas_include_terms = [cfg.tissue]
		#protatlas_include_terms.extend(cfg.seed_include_terms)
		protatlas_include_terms = cfg.seed_include_terms[:]
		protatlas_include_terms.extend(cfg.additional_include_terms)

		self.blockPrint()
		prot_atlas_df, selected_normal_tissues, all_normal_tissues, selected_rna_samples, all_rna_samples = proc_obj.process_protein_atlas_features(protatlas_include_terms, cfg.exclude_terms)
		self.enablePrint()

		if self.verbose:
			print('\nAvailable tissues (normal_tissue.tsv.gz data):')
			print(sorted(all_normal_tissues))

		print('\n' + self.bordered(self.custom_bullet + ' Selected tissues from Protein Atlas (normal_tissue.tsv.gz) ' + self.custom_bullet))
		print(sorted(selected_normal_tissues))

		if self.verbose:
			print('\nAvailable samples (rna_tissue.tsv.gz data):')
			print(sorted(all_rna_samples))

		print('\n' + self.bordered(self.custom_bullet + ' Selected samples from Protein Atlas (rna_tissue.tsv.gz) ' + self.custom_bullet))
		print(sorted(selected_rna_samples))




	def assess_msigdb_filtered_output(self, proc_obj, cfg):
		print(self.line_spacer + "-----------------	Assessing MSigDB filtering [config parameters: 'tissue', 'seed_include_terms', 'additional_include_terms']	-----------------\n")
		#print("- Provided 'tissue':")
		#print(cfg.tissue)
		print("\n- Provided 'seed_include_terms':")
		print(cfg.seed_include_terms)
		print("\n- Provided 'additional_include_terms':")
		print(cfg.additional_include_terms)

		#msigdb_include_terms = [cfg.tissue]
		#msigdb_include_terms.extend(cfg.seed_include_terms)
		msigdb_include_terms = cfg.seed_include_terms[:]
		msigdb_include_terms.extend(cfg.additional_include_terms)
		exclude_terms = cfg.exclude_terms

		if cfg.generic_classifier:
			msigdb_include_terms = ['.*']
			exclude_terms = []

		self.blockPrint()
		msigdb_go_df, selected_go_terms = proc_obj.process_msigdb_go_features(msigdb_include_terms, exclude_terms)
		self.enablePrint()

		print('\n' + self.bordered(self.custom_bullet + ' Selected Gene Ontology terms (from MSigDB) ' + self.custom_bullet))
		print(sorted(selected_go_terms))




	def assess_mgi_filtered_output(self, proc_obj, cfg):
		if cfg.generic_classifier:
			return 0

		print(self.line_spacer + "-----------------	Assessing MGI filtering [config parameters: 'seed_include_terms', 'additional_include_terms']	-----------------\n")
		print("- Provided 'seed_include_terms':")
		print(cfg.seed_include_terms)
		print("\n- Provided 'additional_include_terms':")
		print(cfg.additional_include_terms)

		mgi_include_terms = cfg.seed_include_terms[:]
		mgi_include_terms.extend(cfg.additional_include_terms)

		self.blockPrint()
		_, selected_mgi_phenotypes = proc_obj.process_mgi_features(mgi_include_terms, cfg.exclude_terms)
		self.enablePrint()
		# print(selected_mgi_phenotypes)

		selected_mgi_phenotypes_expanded = [s.split('|') for s in selected_mgi_phenotypes]
		selected_mgi_phenotypes_expanded = list(set([item.lower() for sublist in selected_mgi_phenotypes_expanded for item in sublist]))
		# print(selected_mgi_phenotypes_expanded)

		include_pattern = re.compile('|'.join(mgi_include_terms), re.IGNORECASE)

		filtered_selected_mgi_phenotypes_expanded = list(filter(lambda x: re.search(include_pattern, x), selected_mgi_phenotypes_expanded))
		# print(filtered_selected_mgi_phenotypes_expanded)

		print('\n' + self.bordered(self.custom_bullet + ' Selected MGI phenotypes ' + self.custom_bullet))
		print(sorted(filtered_selected_mgi_phenotypes_expanded))



	def get_english_stopwords(self):
		eng_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

		return eng_stopwords




	def run_mantis_ml_profiler(self):

		print('>>> Running mantis-ml config profiling ...')
		print('verbose:', self.verbose)
		print('Config file:', self.config_file)
		print('Output dir:', self.output_dir)

		cfg = Config(self.config_file, self.output_dir)
		proc_obj = ProcessFeaturesFilteredByDisease(cfg)

		# HPO
		self.assess_hpo_filtered_output(proc_obj, cfg)


		# GTEx
		self.assess_gtex_filtered_output(proc_obj, cfg)


		# Protein Atlas
		self.assess_proteinatlas_filtered_output(proc_obj, cfg)


		# MSigDB
		self.assess_msigdb_filtered_output(proc_obj, cfg)


		# MGI
		self.assess_mgi_filtered_output(proc_obj, cfg)
		print('\n\n<<< mantis-ml config profiling complete.')




	
def main():

	parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
	parser.add_argument("-c", dest="config_file", required=True, help="Config file (.yaml) with run parameters [Required]\n\n")
	parser.add_argument("-o", dest="output_dir", help="Output directory name\n(absolute/relative path e.g. ./CKD, /tmp/Epilepsy-testing, etc.)\nIf it doesn't exist it will automatically be created [Required]\n\n", required=True)
	parser.add_argument('-v', '--verbosity', action="count", help="Print verbose output\n\n")     

	if len(sys.argv)==1:
		parser.print_help(sys.stderr)
		sys.exit(1)

	args = parser.parse_args()      
	config_file = args.config_file
	output_dir = args.output_dir
	verbose = bool(args.verbosity)
	
	profiler = MantisMlProfiler(config_file, output_dir, verbose=verbose)
	profiler.run_mantis_ml_profiler()


if __name__ == '__main__':
	main()
