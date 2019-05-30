import pandas as pd
import sys

df = pd.read_csv('inweb_pairwise_gene_interactions.tsv', sep='\t')
print(df.shape)

rev_df = df[['gene_right', 'gene_left', 'source']]
rev_df.columns = ['gene_left', 'gene_right', 'source']

full_df = pd.concat([df, rev_df], axis=0)
full_df.reset_index(drop=True, inplace=True)
print(full_df.shape)
print(full_df[0:3])
print(full_df[612996:612999])


experim_df = full_df.loc[ full_df['source'] == 'experimental interaction detection', :]
agg_experim_df = pd.DataFrame(experim_df.groupby('gene_left')['gene_right'].apply(list))
agg_experim_df.columns = ['interacting_genes']
agg_experim_df.index.name = 'Gene_Name'
agg_experim_df.reset_index(inplace=True)
print(agg_experim_df.head())
print(agg_experim_df.shape)
agg_experim_df.to_csv('experimental_pairwise_interactions.tsv', sep='\t', index=None)


inferred_df = full_df.loc[ full_df['source'] == 'inference', :]
agg_inferred_df = pd.DataFrame(inferred_df.groupby('gene_left')['gene_right'].apply(list))
agg_inferred_df.columns = ['interacting_genes']
agg_inferred_df.index.name = 'Gene_Name'
agg_inferred_df.reset_index(inplace=True)
print(agg_inferred_df.shape)
print(agg_inferred_df.head())
agg_inferred_df.to_csv('inferred_pairwise_interactions.tsv', sep='\t', index=None)
