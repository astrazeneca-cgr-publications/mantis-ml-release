import pandas as pd

df = pd.read_csv('genemap2.edited.collapsed.txt', sep='\t')
print(df.head())
print(df.shape)

#AD: autosomal dominant
#AR: autosomal recessive

# AD
print('\nAnnotation has AD:')
ad_df = df.loc[df.Annotation.str.contains('AD'), :]
pd.Series(ad_df['Approved_Symbol'].unique()).to_csv('AD_genes.txt', index=False)
print(ad_df.shape)

# AR
print('\nAnnotation has AR:')
ar_df = df.loc[df.Annotation.str.contains('AR'), :]
pd.Series(ar_df['Approved_Symbol'].unique()).to_csv('AR_genes.txt', index=False)
print(ar_df.shape)


# -----------------------------------------------------------------
# AD and not AR
print('\nAnnotation has AD but not AR:')
ad_only_df = df.loc[df.Annotation.str.contains('AD'), :]
ad_only_df = ad_only_df.loc[~df.Annotation.str.contains('AR'), :]
pd.Series(ad_only_df['Approved_Symbol'].unique()).to_csv('AD_only_genes.txt', index=False)
print(ad_only_df.shape)

# AR and not AD
print('\nAnnotation has AR but not AD:')
ar_only_df = df.loc[df.Annotation.str.contains('AR'), :]
ar_only_df = ar_only_df.loc[~df.Annotation.str.contains('AD'), :]
pd.Series(ar_only_df['Approved_Symbol'].unique()).to_csv('AR_only_genes.txt', index=False)
print(ar_only_df.shape)
