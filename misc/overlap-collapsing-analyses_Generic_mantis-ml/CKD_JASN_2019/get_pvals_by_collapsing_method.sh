#!/bin/bash

methods=("v-AURORA-CUMC-all_dom_rare_LOF," "piv-mendelian-all_dom_rare_LOF," "v-AURORA-CUMC-all_dom_ultrarare_OO," "i-all_AURORA_dom_rare_missense," "viii-CUMC-all_dom_rare_mtr50," "v-AURORA-CUMC-all_dom_rare_missense_mtr50," "i-all_AURORA_dom_rare_syn," "piv-mendelian-all_dom_rare_syn," "v-AURORA-CUMC-all_dom_rare_syn," "v-AURORA-CUMC-all-rec_syn," "iii-CUMC-all_dom_rare_syn,")

#collapsing_file="supp_table_S5_Final.csv"
collapsing_file="14-11-17_topG_all_all_models.csv"


for method in ${methods[@]}; do
	echo $method
	stripped_method=${method::-1}
	out_file=CKD/${stripped_method}_collapsing_ranking.CKD.csv
	cat $collapsing_file | grep $method | cut -d',' -f3,14 > $out_file
done
