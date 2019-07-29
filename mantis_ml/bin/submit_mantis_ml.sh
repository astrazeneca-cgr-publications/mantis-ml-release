#!/bin/bash

# ==================================================================================================================================
# ************						Read Input Parameters 						************
# ==================================================================================================================================
# ___ Default values ___
mem=4G #default: 4G; Generic: 12G 
nthreads=10 #default: 30; Generic: 10
time=24:00:00

fixed_nthreads=4

usage="\nUsage: submit_mantis_ml.sh [-h] [-c|--config CONFIG_FILE] [-m|--mem MEMORY]\n\t\t\t\t[-t|--time TIME] [-n|--nthreads NUM_THREADS]"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -c|--config)
        config_file="$2"
        shift; shift
        ;;
        -m|--mem)
        mem="$2"
        shift; shift
        ;;
        -t|--time)
        time="$2"
        shift; shift
        ;;
        -n|--nthreads)
        nthreads="$2"
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
echo "mem: $mem"
echo "nthreads: $nthreads"
echo "time: $time"

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
function wait_for_job() {
    job_name=$1

    cnt=0
    while :
    do
        status=`squeue -u kclc950 -n $job_name`
        status_fields=($status)
        
        if (( $cnt % 10 == 0 )); then 
            echo Waiting for job $job_name ...
            cnt=0
        fi

        # - Running: JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON) 6884828 core CKD_pre kclc950 CG 0:50 1 seskscpn102 
        # - Complete: JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
        if [ "${#status_fields[@]}" -lt 16 ]; then
            break
        fi

        cnt=$((cnt+1))
        sleep 1
    done
}


function run_preprocess_step() {
    # > mantis-ml pre-processing ('pre')
    run_tag=pre
    pre_job_name=${pheno}_$run_tag

    sbatch -J $pre_job_name -o $logs/${pre_job_name}.out.txt -e $logs/${pre_job_name}.err.txt --time=${time} --mem-per-cpu=${mem} --cpus-per-task=${fixed_nthreads} ./mantis_ml_wrapper.sh -c $config_file -r $run_tag
    wait_for_job $pre_job_name
    echo ">> mantis-ml pre-processing ('pre') step complete."
}


function run_boruta_step() {
    # > mantis-ml boruta algorithm ('boruta')
    run_tag=boruta
    boruta_job_name=${pheno}_$run_tag

    sbatch -J $boruta_job_name -o $logs/${boruta_job_name}.out.txt -e $logs/${boruta_job_name}.err.txt --time=${time} --mem-per-cpu=${mem} --cpus-per-task=${nthreads} ./mantis_ml_wrapper.sh -c $config_file -r $run_tag
    echo ">> mantis-ml Boruta ('boruta') step submitted."
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

            sbatch -J $pu_job_name -o $logs/${pu_job_name}.out.txt -e $logs/${pu_job_name}.err.txt --time=${time} --mem-per-cpu=${mem} --cpus-per-task=${nthreads} ./mantis_ml_wrapper.sh -c $config_file -r $run_tag -m $clf_id -s $final_lvl_clf
        else
            sbatch -J $pu_job_name -o $logs/${pu_job_name}.out.txt -e $logs/${pu_job_name}.err.txt --time=${time} --mem-per-cpu=${mem} --cpus-per-task=${nthreads} ./mantis_ml_wrapper.sh -c $config_file -r $run_tag -m $clf_id
        fi
    done
    

    for job in ${all_pu_jobs[@]}; do
        wait_for_job $job
    done
    echo ">> mantis-ml PU Learning ('pu') step complete for all classifiers."
}



function run_postprocess_step() {
    # > mantis-ml post-processing ('post')
    run_tag=post
    post_job_name=${pheno}_$run_tag

    sbatch -J $post_job_name -o $logs/${post_job_name}.out.txt -e $logs/${post_job_name}.err.txt --time=${time} --mem-per-cpu=${mem} --cpus-per-task=${fixed_nthreads} ./mantis_ml_wrapper.sh -c $config_file -r $run_tag
    wait_for_job $post_job_name
    echo ">> mantis-ml post-processing ('post') step complete."
}



function run_post_unsup_step() {
    # > mantis-ml unsupervised learning with annotation from post-processing ('post_unsup')
    run_tag=post_unsup
    post_unsup_job_name=${pheno}_$run_tag

    sbatch -J $post_unsup_job_name -o $logs/${post_unsup_job_name}.out.txt -e $logs/${post_unsup_job_name}.err.txt --time=${time} --mem-per-cpu=${mem} --cpus-per-task=${fixed_nthreads} ./mantis_ml_wrapper.sh -c $config_file -r $run_tag 
    wait_for_job $post_unsup_job_name
    echo ">> mantis-ml post-processing -- unsupervised ('post_unsup') step submitted."
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


printf "\n\n\n\n\n============ Running unsupervised learning with annotation from post-processing results ============\n\n" 
run_post_unsup_step


printf "\n\n\n============ mantis-ml run complete. ============\n"
