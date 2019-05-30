##Source:
[http://nephqtl.org](http://nephqtl.org)

Description:
This is a database of cis-eQTLs of the glomerular and tubulointerstitial tissues of the kidney found in 187 participants in the NEPTUNE cohort.


# Retrieve feature table files
>
cat tub_GeneSummary_FDR_0.01.csv | sed 's/\".*\"//g' | cut -d',' -f2,4,5,6 > tub_feature_table.csv
# edit columns header to: Gene_Name,tub_Exp_num_of_eQTLs,tub_Pr_of_no_eQTL,tub_FDR
