from sys import argv, exit
import pandas as pd

input_file = argv[1]
disease = argv[2]

eqtl_hits_per_gene = dict()

cnt = 0
with open(input_file) as fh:
    for line in fh:
        if cnt == 0:
            cnt += 1
            continue

        line = line.rstrip()
	
        vals = line.split('\t')
        genes = vals[3].split('/')

        for g in genes:
            eqtl_hits_per_gene[g] = eqtl_hits_per_gene.get(g, 0) + 1


df = pd.DataFrame.from_dict(pd.Series(eqtl_hits_per_gene))
df.reset_index(inplace=True)
df.columns = ['Gene_Name', disease + '_eQTL_hits']
print(df.head())

df.to_csv(disease + '_eQTL_hits.csv', index=False)
