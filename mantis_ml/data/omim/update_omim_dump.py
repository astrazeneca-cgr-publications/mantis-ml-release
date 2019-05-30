from sys import argv, exit
import pandas as pd
import urllib.request


print(">> Downloading latest version of OMIM annotation dump file...")
tmp_downl_file = 'genemap2.latest_dump.tmp'
urllib.request.urlretrieve("http://data.omim.org/downloads/KH9-kEspQpy0i0ROx7hPZw/genemap2.txt", tmp_downl_file)
print(".. Download complete: genemap2.latest_dump.tmp\n")

input_file = tmp_downl_file 

# read file except for comments
df = pd.read_csv(input_file, sep='\t', comment='#', header=None)

# retain header line from comments
df_header = ''
with open(input_file) as fh:
    for line in fh:
        if line.startswith('# Chromosome'):
            df_header = line.rstrip()
            break

df_header = df_header.replace('# ', '')
df_cols = df_header.split('\t')
df.columns = df_cols


# ---------------------------------------------------
def splitDataFrameList(df, target_column, separator):
    ''' 
    Expand one-to-many entries into multiple rows based on a delimiter
    
    Input:
        - df (pandas.DataFrame): dataframe to split,
        - target_column (str): the column containing the values to split
        - separator (str): the symbol used to perform the split
    Output:
        a dataframe with each entry for the target column separated, with each element moved into a new row.
        The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row, row_accumulator, target_column, separator):
        split_row = row[target_column].split(separator)

        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows, target_column, separator))
    new_df = pd.DataFrame(new_rows)

    return new_df



print(">> Processing OMIM table...")
# - Exclude rows with NAs in the 'Phenotypes' column
df = df.dropna(subset=['Phenotypes'])

# - Split multiple phenotypes (separated by ';') into multiple rows
df = splitDataFrameList(df, 'Phenotypes', ";")


# delete leading space from Pheynotypes column
df['Phenotypes'] = df['Phenotypes'].str.strip()


# - Retain only entries containing '(3)'
df = df[df['Phenotypes'].str.contains("\\(3\\)")]


# - Drop entries _without_ a Entrez Gene ID value
df = df.dropna(subset=['Entrez Gene ID'])


cols_to_retain = ['Chromosome', 'Mim Number', 'Approved Symbol', 'Entrez Gene ID', 'Ensembl Gene ID', 'Phenotypes']
df = df[cols_to_retain]
print(df.describe())


def add_annotation(entry):
    annot_str = ''
    
    annot_dict = {'Autosomal dominant': 'AD', 
                  'Autosomal recessive': 'AR', 
                  'somatic': 'som', 
                  '[': 'misc', 
                  '{': 'misc', 
                  'X-linked recessive': 'XLR', 
                  'X-linked dominant': 'XLD'}
    
    for key, val in annot_dict.items():
        if key in entry:
            annot_str += val + ';'
    
    if annot_str == '':
        annot_str = 'other'
    else:
        annot_str = annot_str[:-1]
    
    return annot_str

df['Annotation'] = df['Phenotypes'].apply(add_annotation)
df.index = range(df.shape[0])

df.sort_values("Phenotypes", inplace=True)
# df.to_csv('genemap2.edited.txt', sep='\t', index=False)


## Place entries starting with '{', '[' or '?' at the bottom of the data frame
tmp_bottom_entries = df['Phenotypes'].str.startswith(('{', '[', '?'))
bottom_df = df[tmp_bottom_entries]
top_df = df[df['Phenotypes'].str.startswith(('{', '[', '?')) == False]

sorted_df = pd.concat([top_df, bottom_df], axis=0)
print(sorted_df.describe())


print("Aggregating entries with same MIM number...")
def agg_func(x):
    return pd.Series(dict(Approved_Symbol = '|'.join(x['Approved Symbol'].astype(str).unique()),
                         Entrez_Gene_ID = '|'.join(x['Entrez Gene ID'].astype(int).astype(str).unique()),
                         Ensembl_Gene_ID = '|'.join(x['Ensembl Gene ID'].astype(str).unique()),
                         Phenotypes = '|'.join(x['Phenotypes']), 
                         Chromosome = '|'.join(x['Chromosome'].astype(str).unique()), 
                        Annotation = '|'.join(x['Annotation']) ))
    

final_df = sorted_df.groupby('Mim Number').apply(agg_func)
print(final_df.describe())
final_df.insert(loc=0, column='Mim_Number', value=final_df.index)
print(final_df.describe())

# Place entries starting with '{', '[' or '?' at the bottom
tmp_bottom_entries = final_df['Phenotypes'].str.startswith('?')
bottom_df = final_df[tmp_bottom_entries]

tmp_bottom_entries = final_df['Phenotypes'].str.startswith('[')
bottom_df = pd.concat([bottom_df, final_df[tmp_bottom_entries]], axis=0)  

tmp_bottom_entries = final_df['Phenotypes'].str.startswith('{')
bottom_df = pd.concat([bottom_df, final_df[tmp_bottom_entries]], axis=0)  

top_df = final_df[final_df['Phenotypes'].str.startswith(('{', '[', '?')) == False]
final_df = pd.concat([top_df, bottom_df], axis=0)

# replace ',' with ';' to separate different phenotypes within a '|'-separated group of phenotypes
final_df['Phenotypes'] = final_df.Phenotypes.str.replace(',', ';')


# fix bug from original OMIM dump
final_df.loc[final_df.Approved_Symbol == 'GDF1', 'Ensembl_Gene_ID'] = 'ENSG00000130283'


# re-arrange columns order
final_df = final_df[['Chromosome', 'Mim_Number', 'Approved_Symbol', 'Entrez_Gene_ID', 'Ensembl_Gene_ID', 'Phenotypes', 'Annotation']]

output_file = 'genemap2.edited.collapsed.txt'
final_df.to_csv(output_file, sep='\t', index=False)
print(">> [Done] Output file:" + output_file)