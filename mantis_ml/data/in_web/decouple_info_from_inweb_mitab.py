import pandas as pd
import re
import sys

input_file = 'InBio_Map_core_2016_09_12/core.psimitab'

df = pd.read_csv(input_file, sep='\t', header=None) #, nrows=4000)

# drop rows where only one gene is provided - no interaction can be captured
df.dropna(subset=[4], inplace=True)
df.dropna(subset=[5], inplace=True)

cols_of_interest = [4, 5, 6]
df = df[cols_of_interest]

df['gene_left'] = [v[1] for v in df[4].str.split(':|\(', 0).tolist()]
df['gene_right'] = [v[1] for v in df[5].str.split(':|\(', 0).tolist()]
df['source'] = [v[1] for v in df[6].str.split('\(|\)', 0).tolist()]
print(df.head())

df = df[['gene_left', 'gene_right', 'source']]
df.to_csv('inweb_pairwise_gene_interactions.tsv', sep='\t', index=None)
