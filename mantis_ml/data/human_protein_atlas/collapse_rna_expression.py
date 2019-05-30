from sys import argv, exit
import pandas as pd

"""
Add TPM values from lines with same 'Gene_Name'
"""


# all_genes_human_protein_atlas_rna_expression_tpm.tsv.tmp
input_file = argv[1]

df = pd.read_csv(input_file, sep='\t')
print(df.head())

target_col = 'ProteinAtlas_RNA_expression_TMP'


dupl_genes = ['ARL14EPL', 'BTBD8', 'C2orf61', 'COG8', 'HIST1H3D', 'LYNX1', 'MATR3', 'PRSS50', 'RABGEF1', 'SCO2', 'SDHD', 'TXNRD3NB']
print(df.loc[ df.Gene_Name.isin(dupl_genes) ])
df = df.groupby('Gene_Name').sum()
df['Gene_Name'] = df.index.copy()
df = df[['Gene_Name', target_col]]

print(df.loc[ df.Gene_Name.isin(dupl_genes) ])


out_file = input_file.replace('.tmp', '')
df.to_csv(out_file, sep='\t', index=None)
