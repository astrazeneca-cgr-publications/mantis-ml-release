## Get all genes from OMIM with a disease phenotype
```
cat genemap2.edited.collapsed.txt | cut -f3 | sort | uniq | grep -v nan > All_genes.txt
```

__genemap2.edited.collapsed.txt__ has been already processed to contain only genes with _Phenotype_ (3): `the molecular basis for the disorder is known; a mutation has been found in the gene`.
