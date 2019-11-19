import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
from mantis_ml.config_class import Config




class Consensus_Gene_Predictions:

	def __init__(self, config_file, output_dir, top_ratio, gene_class):

		self.config_file = config_file
		self.output_dir = output_dir
		self.top_ratio = top_ratio	 #0.01, 0.05
		self.gene_class = gene_class	# Novel or Known
	
		self.cfg = Config(config_file, self.output_dir)
		self.sorted_classifiers = self.read_sorted_classifers()


		self.color_palette = {'Stacking': '#684392', 'ExtraTreesClassifier': '#35A037', 'SVC': '#651124', 
				      'DNN': '#000000', 'RandomForestClassifier': '#1F7AB9', 
				      'XGBoost': '#F07E21', 'GradientBoostingClassifier': '#E32321'}
		self.clf_alias = {'ExtraTreesClassifier': 'ET', 'SVC': 'SVC', 'DNN': 'DNN', 
				'RandomForestClassifier': 'RF', 'XGBoost': 'XGB', 
				'GradientBoostingClassifier': 'GB', 'Stacking': 'Stacking'}

	def init_dirs(self):
		self.base_enrichment_dir = str(self.cfg.hypergeom_figs_out)
		self.consensus_predictions_dir = str(self.cfg.overlap_gene_predictions)
		if not os.path.exists(self.consensus_predictions_dir):
			os.makedirs(self.consensus_predictions_dir)
	


	def compile_predicted_genes_df(self):

		all_predicted_genes = []
		self.predicted_genes_df = pd.DataFrame()
		for clf in self.sorted_classifiers:
			enrichment_dir = self.base_enrichment_dir + '/' + clf

			tmp_df = pd.read_csv(enrichment_dir + '/' + self.gene_class + '_overlaping_genes.Top_' + str(self.top_ratio) + '.' + self.clf_alias[clf] + '.csv', header=None)
			tmp_predicted_genes = tmp_df.iloc[:, 0].tolist()

			if len(all_predicted_genes) > 0:
				all_predicted_genes = list(set(all_predicted_genes) & set(tmp_predicted_genes))
			else:
				all_predicted_genes = tmp_predicted_genes

			tmp_df[clf] = 1
			tmp_df.index = tmp_predicted_genes
			tmp_df.drop(tmp_df.columns[0], axis=1, inplace=True)
		
			if len(self.predicted_genes_df) > 0:
				self.predicted_genes_df = self.predicted_genes_df.merge(tmp_df, left_index=True, right_index=True, how='outer')  
			else:
				self.predicted_genes_df = tmp_df

		self.predicted_genes_df.fillna(0, inplace=True) 
		self.predicted_genes_df.reset_index(inplace=True)
		self.predicted_genes_df.columns.values[0] = 'Gene_Name'
		#print(self.predicted_genes_df.head())



	def run(self):

		self.init_dirs()
		#try:
		self.compile_predicted_genes_df()

		# The 1st classifier in clf_subset should be the one with best overlap performance
		for relaxation in range(len(self.sorted_classifiers)-1):
			self.get_consensus_list_and_plot(self.predicted_genes_df, self.sorted_classifiers, relaxation=relaxation)


		for top_clf in range(1, len(self.sorted_classifiers)):
			tmp_clf_subset = self.sorted_classifiers[:top_clf]
			self.get_consensus_list_and_plot(self.predicted_genes_df, tmp_clf_subset, top_n_clf=True)
		#except Exception as e:
		#	pass




	def read_sorted_classifers(self):
		"""
		    Read classifiers in descending order of avg. AUC
		"""
		sorted_classifiers = []
		avg_aucs_file = str(self.cfg.superv_out / 'Avg_AUC_per_classifier.txt')
		with open(avg_aucs_file) as fh:
			for line in fh:
				tmp_clf, tmp_auc = line.split('\t')
				sorted_classifiers.append(tmp_clf)

		return sorted_classifiers



	def plot_grouped_barplot(self, df, out_dir, width=0.1, relaxation=0):

		df = df.copy()

		fig, ax = plt.subplots(figsize=(20, 10))
		pos = list(range(df.shape[0]))	

		gene_names = df['Gene_Name']
		df.drop(['Gene_Name'], axis=1, inplace=True)

		for i in range(len(df.columns)):
			clf = df.columns.values[i]
			x_coord = [(p + (width * i)) for p in pos]

			plt.bar(x_coord, df[clf].values, width, alpha=0.5, color=self.color_palette[clf], label=clf)

		ax.set_ylabel('mantis-ml percentile score')
		title_str = 'mantis-ml performance per classifier on the consensus "' + self.gene_class + '" genes from the enrichment test against external ranked gene list'
		title_str += '\n(found in ' + str(len(df.columns) - relaxation) + ' out of ' + str(len(df.columns)) + ' classifiers)'

		ax.set_title(title_str)
		ax.set_xticks([p + (len(df.columns)/4 + 1) * width for p in pos])
		ax.set_xticklabels(gene_names)

		if len(pos) > 20:
			plt.xticks(rotation='vertical')
			

		plt.legend(loc='upper right')
		ax.legend(bbox_to_anchor=(1.05, 1))
		#plt.ylim([95, 100])

		fig.savefig(out_dir + '/' + self.gene_class + '-consensus_predicted_genes.Top_ratio' + str(self.top_ratio) + '.pdf', bbox_inches="tight")




	def find_consensus_from_selected_classifiers(self, clf_subset, df, relaxation=0):

		df = df.copy()
		df['classifier_hits'] = df.sum(axis=1)
		df = df.loc[df['classifier_hits'] >= len(clf_subset) - relaxation, :]
		df.sort_values(by='classifier_hits', ascending=False, inplace=True)

		df = df[['Gene_Name']]
		
		# read gene mantis-ml scores per classifier
		for clf in clf_subset:
			tmp_mantis_ml_file = str(self.cfg.superv_ranked_pred / (clf + '.mantis-ml_predictions.csv'))  
			tmp_mantis_df = pd.read_csv(tmp_mantis_ml_file, header=0)

			if 'known_gene' in tmp_mantis_df.columns:
				tmp_mantis_df.drop(['known_gene'], axis=1, inplace=True)
			tmp_mantis_df = tmp_mantis_df.rename(columns={'mantis_ml_perc': clf}) 
			tmp_mantis_df.drop(['mantis_ml_proba'], axis=1, inplace=True)

			df = df.merge(tmp_mantis_df, left_on='Gene_Name', right_on='Gene_Name', how='left')

		top_clf = self.sorted_classifiers[0]
		if top_clf not in df.columns:
			top_clf = df.columns.values[0]
		df.sort_values(by=top_clf, inplace=True, ascending=False)	

		# Sort by sum of prediction proba across all classifiers
		#tmp_df = df.drop(['Gene_Name'], axis=1)
		#tmp_df['sum'] = tmp_df.sum(axis=1)
		#tmp_df['Gene_Name'] = df['Gene_Name']
		#df = tmp_df.sort_values(by='sum', ascending=False)
		#df.drop(['sum'], axis=1, inplace=True)

		df.reset_index(drop=True, inplace=True)

		return df 



	def get_consensus_list_and_plot(self, df, clf_subset, relaxation=0, width=0.1, top_n_clf=False):

		if len(clf_subset) - relaxation < 1:
			relaxation = 0

		# Create output dir for consensus gene lists and plots
		if len(clf_subset) == len(self.sorted_classifiers):
			if relaxation == 0:
				out_dir = 'All_classifiers'
			else:
				out_dir = str(len(clf_subset) - relaxation) + '_out_of_' + str(len(clf_subset)) + '_classifiers'
		else:
			if top_n_clf:
				out_dir = 'Top-' + str(len(clf_subset)) + '_classifiers'
			else:
				out_dir = '_'.join([self.clf_alias[c] for c in clf_subset])
		out_dir = 'Predicted_by-' + out_dir
		out_dir = self.consensus_predictions_dir + '/' + out_dir

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)


		cons_df = self.find_consensus_from_selected_classifiers(clf_subset, df, relaxation=relaxation)
		self.plot_grouped_barplot(cons_df, out_dir, relaxation=relaxation, width=width)

		print('\n> ' + ', '.join(clf_subset) + ':')

		cons_df.to_csv(out_dir + '/' + self.gene_class + '-consensus_predicted_genes.Top_ratio' + str(self.top_ratio) + '.tsv', index=False, sep='\t')

		consensus_predicted_genes = cons_df['Gene_Name'].values
		print(consensus_predicted_genes)
		with open(out_dir + '/' + self.gene_class + '-consensus_predicted_genes.Sorted_by_top_classifier.Top_ratio' + str(self.top_ratio) + '.txt', 'w') as fh:
			for g in consensus_predicted_genes:
				fh.write(g + '\n')




if __name__ == '__main__':

	config_file = sys.argv[1]
	output_dir = sys.argv[2]
	top_ratio = sys.argv[3]	#0.01, 0.05
	gene_class = sys.argv[4]	# Novel or Known
	print('\ngene class:', gene_class, '\ntop_ratio:', top_ratio)


	cons_obj = Consensus_Gene_Predictions(config_file, output_dir, top_ratio, gene_class)
	cons_obj.run()
