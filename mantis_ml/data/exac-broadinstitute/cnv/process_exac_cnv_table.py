import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 80)



df = pd.read_csv('exac-final-cnv.gene.scores071316', sep=' ')
df.head()

df.drop(['gene', 'chr', 'start', 'end'], axis=1, inplace=True)
df.head()

agg_df = df.groupby('gene_symbol').mean()

# Make sure flags in the interval 0<=flag<0.5 are assigned a '0' value
# and flags between 0.5<=flag<=1 a '1' value, after groupping by mean.
agg_df.loc[ (agg_df.flag >= 0.5), 'flag' ] = 1
agg_df.loc[ (agg_df.flag < 0.5), 'flag' ] = 0

agg_df.columns = 'ExAC_' + agg_df.columns
agg_df.insert(0, 'Gene_Name', agg_df.index)
agg_df.head()

agg_df.to_csv('ExAC_CNV_features.tsv', sep='\t', index=None)
