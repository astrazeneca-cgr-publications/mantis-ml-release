#!/bin/bash 

# ==== Define default arg values ====
clf_id=None
final_lvl_classifier=None

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -c|--config)
        config_file="$2"
        shift; shift
        ;;
        -m|--model)
        clf_id="$2"
        shift; shift
        ;;
        -s|--stacking)
        final_lvl_classifier="$2"
        shift; shift
        ;;
        -r|--run)
        run_tag="$2"
        shift; shift
        ;;
esac
done
set -- "${POSITIONAL[@]}"
   
echo "config_file: $config_file"
echo "clf_id: $clf_id"
echo "final_lvl_classifier: $final_lvl_classifier"
echo "run_tag: $run_tag"

# Run main mantis-ml script
python -u main.py -c $config_file -r $run_tag -m $clf_id -s $final_lvl_classifier
