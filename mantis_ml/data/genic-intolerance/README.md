Sources:
wget http://genic-intolerance.org/data/GenicIntolerance_v3_12Mar16.txt


# Processing:
cat GenicIntolerance_v3_12Mar16.txt | cut -f1,5 > all_genes_RVIS.tsv  [##Header: Gene_Name	RVIS]
cat GenicIntolerance_v3_12Mar16.txt | cut -f1,24 > all_genes_LoF_FDR_ExAC.tsv  [##Header: Gene_Name	LoF_FDR_ExAC]

cat RVIS_Unpublished_ExAC_May2015.txt | cut -f1,7 > all_genes_RVIS_ExAC.tsv  [##Header: Gene_Name       RVIS_ExAC]

cat RVIS_Unpublished_ExACv2_March2017.txt | cut -f1,2,4,7 > all_genes_RVIS_MTR_ExACv2.tsv  [##Header: Gene_Name       geneCov_ExACv2  RVIS_ExACv2	MTR_ExACv2]
