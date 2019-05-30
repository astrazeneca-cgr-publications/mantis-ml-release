import pandas as pd

## MGI_GenePheno.rpt
mgi_gene_pheno_df = pd.read_csv('MGI_GenePheno.rpt', sep='\t', header=None)
mgi_gene_pheno_df.columns = ['Allelic Composition', 'Allele Symbol(s)', 'Allele ID(s)', 'Genetic Background', 
                             'Mammalian Phenotype ID', 'PubMed ID (pipe-delimited)', 'MGI Marker Accession ID (pipe-delimited)',
                             'MGI Genotype Accession ID (pipe-delimited)']
print(mgi_gene_pheno_df.head())


## HMD_HumanPhenotype.rpt
hmd_human_pheno_df = pd.read_csv('HMD_HumanPhenotype.rpt', sep='\t', header=None)
hmd_human_pheno_df.columns = ['Human Marker Symbol', 'Human Entrez Gene ID', 'HomoloGene ID', 'HGNC Association?',
                              'Mouse Marker Symbol', 'MGI Marker Accession ID', 
                              'High-level Mammalian Phenotype ID (space-delimited)', 'temp']
hmd_human_pheno_df.drop('temp', axis=1, inplace=True)
print(hmd_human_pheno_df.head())


## VOC_MammalianPhenotype.rpt
voc_mammalian_pheno_df = pd.read_csv('VOC_MammalianPhenotype.rpt', sep='\t', header=None)
voc_mammalian_pheno_df.columns = ['Mammalian Phenotype ID', 'Mammalian Phenotype', 'Description']
voc_mammalian_pheno_df.drop(['Description'], axis=1, inplace=True)
voc_mammalian_pheno_df.head()




combined_mgi_hmd_df = pd.merge(mgi_gene_pheno_df, hmd_human_pheno_df, how='outer', 
                               left_on='MGI Marker Accession ID (pipe-delimited)',
                               right_on='MGI Marker Accession ID')


#print(combined_mgi_hmd_df.shape)
#print(combined_mgi_hmd_df.head())



combined_mgi_voc = pd.merge(mgi_gene_pheno_df, voc_mammalian_pheno_df, how='outer', left_on='Mammalian Phenotype ID', 
                       right_on='Mammalian Phenotype ID')
#print(combined_mgi_voc.head())


# For each entry in `HMD_HumanPhenotype.rpt`:
# ---- append the linked phenotype field names (field: __mammalian phenotype__) from `VOC_MammalianPhenotype.rpt`  
# ---- based on matching of __High-level Mammalian Phenotype ID__ and __MP:0000001__ fields between the two files.
def map_phenotype_ids_to_strings(row):
    mp_ids = [x for x in str(row).split(' ') if x != '']
#     print(mp_ids)
    
    cur_phenotypes_series = voc_mammalian_pheno_df.loc[ voc_mammalian_pheno_df['Mammalian Phenotype ID'].isin(mp_ids), 'Mammalian Phenotype']
    if cur_phenotypes_series.shape[0] > 0:
        return '|'.join(cur_phenotypes_series.values)
    else:
        return ''

phenotypes_per_gene = hmd_human_pheno_df['High-level Mammalian Phenotype ID (space-delimited)'].apply(map_phenotype_ids_to_strings)
phenotypes_per_gene.name = 'human_phenotypes'
hmd_human_pheno_df = pd.concat([hmd_human_pheno_df, phenotypes_per_gene], axis=1)

out_file = 'hmd_human_pheno.processed.rpt'
hmd_human_pheno_df.to_csv(out_file, sep='\t', index=None)
print("Saved output to '{0}'".format(out_file))
