
# coding: utf-8

# In[98]:


import matplotlib 
#matplotlib.use('agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats.stats import pearsonr
import sys
import mantis_ml.config as cfg

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# dataset = sys.argv[1]
dataset = 'GGE'


# read primary analysis collapsing results
prim_df = pd.read_csv(dataset + '/primary_collapsing_ranking_epilepsy.' + dataset + '.csv', header=None)
prim_df.columns = ['Gene_Name', 'pval']
prim_df.sort_values(by='pval', ascending=True, inplace=True)
prim_df.reset_index(drop=True, inplace=True)
print(prim_df.head())


# read synonymous variants collapsing results
syn_df = pd.read_csv(dataset + '/synonymous_collapsing_ranking_epilepsy.' + dataset + '.csv', header=None)
syn_df.columns = ['Gene_Name', 'pval']
syn_df.sort_values(by='pval', ascending=True, inplace=True)
syn_df.reset_index(drop=True, inplace=True)
print(syn_df.head())


merged_df = pd.merge(prim_df, syn_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
print(merged_df.head())

print('P-val correlation between primary & synonymous:', pearsonr(merged_df['pval_x'], merged_df['pval_y']))




syn_df.hist(figsize=(7, 7))
plt.title('Synonymous')
plt.savefig(dataset + '/synon_variant_hist.pdf', bbox_inches='tight')
prim_df.hist(figsize=(7, 7))
plt.title('Primary')
plt.savefig(dataset + '/primary_variant_hist.pdf', bbox_inches='tight')


syn_df['x'] = syn_df.index.values + 1
prim_df['x'] = prim_df.index.values + 1


syn_df.plot.line('x', 'pval', figsize=(8, 5), title='Synonymous')
#plt.savefig(dataset + '/synon_variant_lineplot.pdf', bbox_inches='tight')

prim_df.plot.line('x', 'pval', figsize=(8, 5), title='Primary')
plt.savefig(dataset + '/primary_variant_lineplot.pdf', bbox_inches='tight')

# TO-DO: calc correlation between primary and synonymous p-values


# In[84]:


syn_df.drop('x', axis=1, inplace=True)
print(syn_df.head())
proc_feat_df = pd.read_csv(cfg.processed_data_dir / 'processed_feature_table.tsv', sep='\t')
proc_feat_df.head()

original_syn_df = syn_df.copy()


# In[132]:


from matplotlib.colors import ListedColormap


def get_pearsonr_per_feature(df, pval_cutoff=1, plot_hist=True, analysis_type='primary'):

    df = df.loc[ df.pval < pval_cutoff, :]
    
    pearson_corr_per_feature = {}
    pearson_pval_per_feature = {}

    feature_cols = [f for f in proc_feat_df.columns.values if f != 'Gene_Name']

    for f in feature_cols:
        tmp_df = proc_feat_df[['Gene_Name', f]]

        tmp_merged_df = pd.merge(df, tmp_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
        tmp_merged_df.fillna(0, inplace=True)


        pearson_corr, pearson_pval = pearsonr(tmp_merged_df['pval'], tmp_merged_df[f])
        pearson_corr_per_feature[f] = pearson_corr
        pearson_pval_per_feature[f] = pearson_pval


    corr_df = pd.Series(pearson_corr_per_feature).to_frame()
    corr_df.reset_index(inplace=True)
    corr_df.columns = ['feature', 'pearsonr']

    pval_df = pd.Series(pearson_pval_per_feature).to_frame()
    pval_df.reset_index(inplace=True)
    pval_df.columns = ['feature', 'corr_pval']

    full_corr_df = pd.merge(corr_df, pval_df, how='left', left_on='feature', right_on='feature')
    full_corr_df.sort_values(by='pearsonr', inplace=True, ascending=False)
    full_corr_df.head()
    full_corr_df.tail()

    if plot_hist:
        _ = full_corr_df.hist(column='pearsonr', figsize=(10, 7), grid=False)

    return full_corr_df


colors = sns.diverging_palette(10, 220, sep=80, n=len(full_corr_df)).as_hex()
colors = colors[::-1]
sns.set_palette(colors)





analysis_type = 'synonymous' #'primary' #synonymous

if analysis_type == 'primary':
    df = prim_df.copy()
else:
    df = syn_df.copy() 


full_corr_df = get_pearsonr_per_feature(df, analysis_type=analysis_type)
title = '[' + analysis_type + ' analysis] Keep only genes with pval < 1'
_ = full_corr_df.plot.barh(x='feature', y='pearsonr', figsize=(16, 18), grid=False, title=title, fontsize=14)
_ = plt.xlabel("Pearsons'r with collapsing " + analysis_type + " p-values", fontsize=14)
plt.savefig(dataset + '/Correlations.' + analysis_type + '_pvals_vs_features.pval_lt_1.pdf', bbox_inches='tight')


full_corr_df = get_pearsonr_per_feature(df, pval_cutoff=1.1, plot_hist=False, analysis_type=analysis_type)
title = '[' + analysis_type + ' analysis] Keep all genes'
_ = full_corr_df.plot.barh(x='feature', y='pearsonr', figsize=(16, 18), grid=False, title=title, fontsize=14)
_ = plt.xlabel("Pearsons'r with collapsing " + analysis_type + " p-values", fontsize=14)
plt.savefig(dataset + '/Correlations.' + analysis_type + '_pvals_vs_features.all.pdf', bbox_inches='tight')

