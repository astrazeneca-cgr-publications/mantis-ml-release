## Source:
https://www.proteinatlas.org/about/download (last accessed 9 Nov 2018)

## Pre-processing

# -- CKD-specific
> Normal tissue 
zcat < normal_tissue.tsv.gz | grep -E "Gene name|kidney" | cut -f2,3,4,5,6 | grep -v 'Uncertain' | cut -f1,3,4 | sed 's/Gene name/Gene_Name/' | sed 's/ /_/' | sed 's/cells_in //' | cut -f1,2,3 | sed 's/Level/protein_atlas_gene_expr_in_tissue/' > all_genes_human_protein_atlas_kidney_expression.tsv.tmp

python collapse_normal_tissue_expression.py all_genes_human_protein_atlas_kidney_expression.tsv.tmp



> RNA tissue
zcat < rna_tissue.tsv.gz | grep -E "Gene name|kidney" | cut -f2,4 | sed 's/Gene name/Gene_Name/' | sed 's/Value/ProteinAtlas_RNA_expression_TMP/' > all_genes_human_protein_atlas_rna_expression_tpm.tsv.tmp

python collapse_rna_expression.py all_genes_human_protein_atlas_rna_expression_tpm.tsv.tmp



## TO-DO:
Needs further processing to collapse lines with the same 'Gene_Name'
