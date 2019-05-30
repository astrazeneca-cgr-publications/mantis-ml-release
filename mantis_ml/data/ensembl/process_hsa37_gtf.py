import sys
import gzip

# Warning: you need to download 'Homo_sapiens.GRCh37.87.gtf.gz' first in the current directory

input_file = 'Homo_sapiens.GRCh37.87.gtf.gz'
out_fh = gzip.open('hgnc_to_ens_gids.txt.gz', 'w')


hgnc_to_ens_ids = dict()

cnt = 0
with gzip.open(input_file) as fh:
	for line in fh:
		line = line.decode('utf-8').rstrip()
		if line.startswith('#'):
			continue

		vals = line.split('\t')
		annot = vals[8]

		dd = dict(s.strip().replace('"', '').split(' ') for s in annot.split(';') if s != '')

		gene_id = dd['gene_id']
		gene_name = dd['gene_name']

		#print(gene_name, gene_id)

		if gene_name in hgnc_to_ens_ids:
			hgnc_to_ens_ids[gene_name].append(gene_id)
			hgnc_to_ens_ids[gene_name] = list(set(hgnc_to_ens_ids[gene_name]))
		else:
			hgnc_to_ens_ids[gene_name] = [gene_id]
		#print(hgnc_to_ens_ids)

		#cnt += 1
		#if cnt == 2:
		#	sys.exit()
print(len(hgnc_to_ens_ids))
#print(hgnc_to_ens_ids)

for k, v in hgnc_to_ens_ids.items():
	out_fh.write(str.encode(k + '\t' + ''.join(v) + '\n'))
out_fh.close()

