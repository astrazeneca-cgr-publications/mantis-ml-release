import pandas as pd
import re
import numpy as np
from sys import exit
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



def explode(df, column, sep='|', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df




input_file = 'all_associations_v1.0.2.tsv'

full_df = pd.read_csv(input_file, sep='\t', low_memory=False)
full_df.head(2)



# disease_search_term = 'Kidney'
hgnc_genes_series = pd.read_csv('../exac-broadinstitute/all_hgnc_genes.txt', header=None).loc[:, 0]
# hgnc_genes_series[:10]




def get_gwas_features(df, pattern='', search_term='All'):
    print('Genes to keep:', search_term)
    df = df[ df['DISEASE/TRAIT'].str.contains(pattern)]

    # Keep only hits that achieved genome-wide significance
    signif_thres = '5e-08'
    signif_df = df.loc[ df['P-VALUE'].astype(float) < float(signif_thres) ]
    signif_df = signif_df[ ['REPORTED GENE(S)', 'P-VALUE', 'DISEASE/TRAIT', 'MAPPED_TRAIT'] ]
    # signif_df.head()

    # Split comma-separated Genes into separate lines
    expanded_df = explode(signif_df, 'REPORTED GENE(S)', sep=',')
    expanded_df.head()


    # Count hits per gene
    hits_per_gene_df = expanded_df.groufpby('REPORTED GENE(S)').agg('count')

    # Limit to HGNC genes
    hits_per_gene_df = hits_per_gene_df.reindex(hgnc_genes_series)
    hits_per_gene_df.fillna(0, inplace=True)
    hits_per_gene_df.sort_values(by='P-VALUE', inplace=True, ascending=False)
    hits_per_gene_df.reset_index(inplace=True)
    hits_per_gene_df.columns.values[[0, 1]] = ['Gene_Name', 'GWAS_hits']
    hits_per_gene_df = hits_per_gene_df.iloc[:, [0,1]]
    hits_per_gene_df.head()
    hits_per_gene_df.tail()
    hits_per_gene_df.shape


    # --------------------
    # Get min and max P-value for each gene
    pval_df = hits_per_gene_df.merge(expanded_df, how='left', left_on='Gene_Name', right_on='REPORTED GENE(S)')
    pval_df['P-VALUE'] = pval_df['P-VALUE'].apply(lambda x: float(x))
    pval_df.head()
    pval_df.info()

    expanded_df.loc[ expanded_df['REPORTED GENE(S)'] == 'FADS1' ].head()


    max_pval_by_gene = pd.DataFrame(pval_df.groupby('Gene_Name')['P-VALUE'].agg('max'))
    max_pval_by_gene = max_pval_by_gene.rename( columns = {'P-VALUE': 'GWAS_max_P_value'}).reset_index()
    max_pval_by_gene.fillna(1, inplace=True)
    max_pval_by_gene.head()
    print(max_pval_by_gene.loc[ max_pval_by_gene.Gene_Name == 'FADS1'])

    min_pval_by_gene = pd.DataFrame(pval_df.groupby('Gene_Name')['P-VALUE'].agg('min'))
    min_pval_by_gene = min_pval_by_gene.rename( columns = {'P-VALUE': 'GWAS_min_P_value'}).reset_index()
    min_pval_by_gene.fillna(1, inplace=True)
    min_pval_by_gene.head()
    min_pval_by_gene.shape
    print(min_pval_by_gene.loc[ min_pval_by_gene.Gene_Name == 'FADS1'])

    # --------------------

    conc_df = hits_per_gene_df.merge(max_pval_by_gene, how='left', left_on='Gene_Name', right_on='Gene_Name')
    conc_df = conc_df.merge(min_pval_by_gene, how='left', left_on='Gene_Name', right_on='Gene_Name')

    conc_df['GWAS_tissue_trait_flag'] = None
    conc_df.loc[ conc_df.GWAS_hits > 0, 'GWAS_tissue_trait_flag' ] = 1
    conc_df.loc[ conc_df.GWAS_hits == 0, 'GWAS_tissue_trait_flag' ] = 0
    print(conc_df.loc[ conc_df.GWAS_tissue_trait_flag == 1 ].shape)
    if search_term != 'All':
        conc_df.columns = [ s.replace('GWAS', search_term + '_GWAS') for s in conc_df.columns.values ]
    
    conc_df.to_csv(search_term + '_genes_GWAS_features.tsv', sep='\t', index=None)
    print(conc_df.head())

    
## Get GWAS hits for all traits
get_gwas_features(full_df)

print('----------------------')

## Filter by Disease trait
# - CKD
pattern = re.compile('kidney|[^(ad)]renal|nephro|eGFR|dialysis', re.IGNORECASE)
get_gwas_features(full_df, pattern=pattern, search_term='Kidney')

