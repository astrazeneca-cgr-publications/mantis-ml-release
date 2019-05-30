import pandas as pd

input_file = '1-s2.0-S0002929716300428-mmc3.csv'

df = pd.read_csv(input_file)
df.eGene = df.eGene.str.replace('*', '')
print(df.head())

grouped_df = pd.DataFrame(df.groupby(by='eGene')['eQTL_pvalue'].count())

grouped_df.reset_index(inplace=True)
grouped_df.columns = ['Gene_Name', 'platelets_eQTL']
print(grouped_df.head()) 

grouped_df.to_csv('platelets_eQTL.csv', index=None)
