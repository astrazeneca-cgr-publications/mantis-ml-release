from sys import argv, exit
import pandas as pd

"""
# Category coding for aggregation across multiple cell lines in same tissue
- Not detected: 0
- Low: 1
- Medium: 3
- High: 8

Rule: Keeping the max. value after aggregation by same Gene Name
"""


# all_genes_human_protein_atlas_kidney_expression.tsv.tmp
input_file = argv[1]

df = pd.read_csv(input_file, sep='\t')
print(df.head())

target_col = 'protein_atlas_gene_expr_in_tissue'


df.replace( {target_col: {'Not detected': 0, 'Low': 1, 'Medium': 3, 'High': 8}}, inplace=True)
print(df.head())

df[target_col] = df[target_col].astype(str)



tmp_df = pd.DataFrame(df.groupby('Gene_Name')[target_col].agg('|'.join))

tmp_df['delim_cnt'] = tmp_df[target_col].apply(lambda x: x.count('|'))
tmp_df['final_level'] = tmp_df[target_col].apply(lambda x: max(x.split('|')))
tmp_df['Gene_Name'] = tmp_df.index.copy()
print(tmp_df.head())
print(tmp_df.loc[ tmp_df['delim_cnt'] > 2, : ])


final_df = tmp_df[ ['Gene_Name', 'final_level'] ].copy()
final_df.rename( columns={'final_level': target_col }, inplace=True)
print(final_df.head())
final_df.replace( {target_col: {'0': 'Not_detected', '1': 'Low', '3': 'Medium', '8': 'High'}}, inplace=True)
print(final_df.head())


out_file = input_file.replace('.tmp', '')
final_df.to_csv(out_file, sep='\t', index=None)
