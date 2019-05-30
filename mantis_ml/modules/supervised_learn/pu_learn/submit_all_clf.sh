#!/bin/sh

# === read phenotype to append in job IDs ===
if [ "$#" -ne 1 ]; then
	echo -e "[Error]: No input argument was defined.\nPlease define Job_ID string for current submission (e.g. CKD, IPF, etc.)\n"
	exit
fi
pheno=$1
echo "Job ID: ${pheno}"
# ===========================================

#classifiers=(ExtraTreesClassifier RandomForestClassifier GradientBoostingClassifier SVC XGBoost DNN Stacking)
classifiers=(ExtraTreesClassifier RandomForestClassifier DNN SVC)
mem=4G #4G #12G 
cores=30 #30 #10
time=24:00:00


for clf_id in ${classifiers[@]}; do
	if [ $clf_id = 'Stacking' ]
	then
		for final_lvl_clf in DNN; do
			job_id=${clf_id}_${final_lvl_clf}

			echo -e "\n> $job_id:"
			echo "sbatch -J ${pheno}_$job_id -o ${pheno}_${job_id}_out.txt -e ${pheno}_${job_id}_err.txt --time=${time} --mem-per-cpu=${mem} --cpus-per-task=${cores} ./run_pu.sh $clf_id $final_lvl_clf"
			sbatch -J ${pheno}_$job_id -o ${pheno}_${job_id}_out.txt -e ${pheno}_${job_id}_err.txt --time=${time} --mem-per-cpu=${mem} --cpus-per-task=${cores} ./run_pu.sh $clf_id $final_lvl_clf
		done
	else
		echo -e "\n> $clf_id:"
		echo "sbatch -J ${pheno}_${clf_id} -o ${pheno}_${clf_id}_out.txt -e ${pheno}_${clf_id}_err.txt --time=${time} --mem-per-cpu=${mem} --cpus-per-task=${cores} ./run_pu.sh $clf_id"
		sbatch -J ${pheno}_${clf_id} -o ${pheno}_${clf_id}_out.txt -e ${pheno}_${clf_id}_err.txt --time=${time} --mem-per-cpu=${mem} --cpus-per-task=${cores} ./run_pu.sh $clf_id
	fi
done
