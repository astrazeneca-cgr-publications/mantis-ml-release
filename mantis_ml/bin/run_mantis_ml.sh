#!/bin/bash

# ==================================================================================================================================
# ************						Read Input Parameters 						************
# ==================================================================================================================================

usage="\nUsage: run_mantis_ml.sh [-h] [-c|--config CONFIG_FILE]"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -c|--config)
        config_file="$2"
        shift; shift
        ;;
        -h)
        echo -e $usage; exit
        shift; shift
        ;;
	esac
done
set -- "${POSITIONAL[@]}"

echo "config_file: $config_file"

if [ -z "$config_file" ]; then
        echo -e $usage
	echo -e "\n[Error]: No 'config.yaml' file was provided. Please rerun using option: -c|--config."
	exit
fi
# ---------------------------------------------------------------------------------------------------------------------------------



# ==================================================================================================================================
# ************				Extract run-specific Parameters from config.yaml				************
# ==================================================================================================================================
# ____________ Read specified **Classifiers** from input config file ____________
conf_clf=`cat $config_file | grep classifiers | awk -F'[:#]' '{print $2}' | sed -e 's/\[//g' -e 's/\]//g' | tr -d '\040\011\012\015'`
IFS=', ' read -r -a classifiers <<< "$conf_clf"

echo '> Classifiers:'
for clf in "${classifiers[@]}"; do
	echo $clf
done

run_boruta=`cat $config_file | grep run_boruta | awk -F'[:#]' '{print $2}'`
echo "> Run Boruta: $run_boruta"

run_unsupervised=`cat $config_file | grep run_unsupervised | awk -F'[:#]' '{print $2}'`
echo "> Run unsupervised (PCA, t-SNE and UMAP): $run_unsupervised"

# ____________ Read specified **Phenotype** from input config file ____________
pheno=`cat $config_file | grep phenotype | awk -F'[:#]' '{print $2}' | tr -d '\040\011\012\015'`
echo "> Phenotype: ${pheno}"

# ____________ Read specified **run_id** from input config file ____________
run_id=`cat $config_file | grep run_id | awk -F'[:#]' '{print $2}' | tr -d '\040\011\012\015'`

# ____________ Creat logs dir for current phenotype ____________
logs_dir=logs
mkdir -p $logs_dir
logs="$logs_dir/${pheno}-${run_id}"
mkdir -p $logs
# ----------------------------------------------------------------------------------------------------------------------------------





# ==================================================================================================================================
# ****  				                	Functions				                        ****
# ==================================================================================================================================
function run_preprocess_step() {
    # > mantis-ml pre-processing ('pre')
    run_tag=pre
    pre_job_name=${pheno}_$run_tag
    ./mantis_ml_wrapper.sh -c $config_file -r $run_tag

    echo ">> mantis-ml pre-processing ('pre') step complete."
}


function run_boruta_step() {
    # > mantis-ml boruta algorithm ('boruta')
    run_tag=boruta
    boruta_job_name=${pheno}_$run_tag
    ./mantis_ml_wrapper.sh -c $config_file -r $run_tag

    echo ">> mantis-ml Boruta ('boruta') step complete."
}


function run_pu_step() {
    # > mantis-ml pu learning for each classifier('pu')
    run_tag=pu
    all_pu_jobs=()

    for clf_id in ${classifiers[@]}; do
        pu_job_name=${pheno}_${run_tag}_${clf_id}
        all_pu_jobs+=($pu_job_name)
    
        if [ $clf_id = 'Stacking' ]; then
            final_lvl_clf=DNN
            job_id=${clf_id}_${final_lvl_clf}
            ./mantis_ml_wrapper.sh -c $config_file -r $run_tag -m $clf_id -s $final_lvl_clf
        else
            ./mantis_ml_wrapper.sh -c $config_file -r $run_tag -m $clf_id
        fi
    done
    
    echo ">> mantis-ml PU Learning ('pu') step complete for all classifiers."
}



function run_postprocess_step() {
    # > mantis-ml post-processing ('post')
    run_tag=post
    post_job_name=${pheno}_$run_tag
    ./mantis_ml_wrapper.sh -c $config_file -r $run_tag

    echo ">> mantis-ml post-processing ('post') step complete."
}



function run_post_unsup_step() {
    # > mantis-ml unsupervised learning with annotation from post-processing ('post_unsup')
    clf_id=$1
    run_tag=post_unsup
    post_unsup_job_name=${pheno}_$run_tag
    ./mantis_ml_wrapper.sh -c $config_file -r $run_tag -m $clf_id

    echo ">> mantis-ml post-processing -- unsupervised ('post_unsup') step complete."
}

# ----------------------------------------------------------------------------------------------------------------------------------




# ========================================================
# ****                  Main Workflow                 ****
# ========================================================
printf "\n\n\n\n\n============ Running pre-processing step (feature compilation, filtering, EDA, etc.) ============\n\n"
run_preprocess_step


if [ $run_boruta == "True" ]; then
	printf "\n\n\n\n\n============ Running feature importance estimation with Boruta algorithm ============\n\n"
	run_boruta_step
fi


printf "\n\n\n\n\n============ Running stochastic Positive-Unlabelled Learning for each classifier ============\n\n"
run_pu_step


printf "\n\n\n\n\n============ Running post-processing step (results aggregation) ============\n\n"
run_postprocess_step


printf "\n\n\n============ mantis-ml run complete. ============\n"
