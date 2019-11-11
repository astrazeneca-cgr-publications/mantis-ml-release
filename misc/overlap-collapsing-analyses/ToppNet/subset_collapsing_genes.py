import pandas as pd
import sys

phenotype = sys.argv[1]	# e.g. #'CKD', 'ALS', 'Epilepsy'


collapsing_files_dict = {'CKD': "../CKD_JASN_2019/CKD/iii-CUMC-all_dom_rare_syn_collapsing_ranking.CKD.csv", 'ALS': "../ALS_Science_2015/ALS/Dom_LoF_collapsing_ranking.ALS.csv", 'Epilepsy': "../Epilepsy-LancetNeurology_2017/GGE/lof_collapsing_ranking.GGE.csv"}

toppnet_file = phenotype + ".toppnet.ranked_genes.txt"
collapsing_file = collapsing_files_dict[phenotype]


collapsing_df = pd.read_csv(collapsing_file, header=None)
collapsing_df.columns = ['gene', 'ranking']
collapsing_df.gene = collapsing_df.gene.str.replace("'", "")
print(collapsing_df.head())
collapsing_genes = collapsing_df.iloc[:, 0].values.tolist()
print(collapsing_genes[:10])
print(len(collapsing_genes))


toppnet_df = pd.read_csv(toppnet_file, header=None)
toppnet_df.columns = ['gene', 'ranking']
print(toppnet_df.head())
toppnet_genes = toppnet_df.iloc[:, 0].values.tolist()
print(toppnet_genes[:10])
print(len(toppnet_genes))


genes_intersection = list(set(collapsing_genes) & set(toppnet_genes))
print('\nGenes intersection:', len(genes_intersection))
print(genes_intersection[:10])



toppnet_subset_df = toppnet_df.loc[ toppnet_df.gene.isin(genes_intersection), :]
print(toppnet_subset_df.head())
print(toppnet_subset_df.shape)
toppnet_subset_df.to_csv(toppnet_file + '.collapsing_intersection', index=False, header=False)

with open('collapsing_genes_intersection.' + phenotype + '.txt', 'w') as fh:
	for gene in genes_intersection:
		fh.write(gene + '\n')


