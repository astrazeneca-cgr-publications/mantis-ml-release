#!/bin/bash

test_err_file="test_installation.err"
rm -f $test_err_file
test_out_file="test_installation.out"
rm -f $test_out_file



function check_test_err {

	test_num=$1
	module=$2


	num_lines=`cat $test_err_file | wc -l`

	#if [ -s $test_err_file ]; then
	if [ "$num_lines" -gt 1 ]; then
		echo -e "[$module run]\tTest ${test_num}/3......FAILED - Aborting..."
		echo "Please check the $test_err_file file for more details"
		exit
	else
		echo -e "[$module run]\tTest ${test_num}/3......OK"
	fi

}



mantisml-profiler -c mantis_ml/conf/CKD_config.yaml -o ckd-test 1>$test_out_file 2>$test_err_file
check_test_err 1 "profiler"


mantisml -c mantis_ml/conf/CKD_config.yaml -o ckd-test -r pre 1>>$test_out_file 2>$test_err_file
check_test_err 2 "pre-processing"
		

mantisml -c mantis_ml/conf/CKD_config.yaml -o ckd-test -r pu -m dnn -i 1 1>>$test_out_file 2>$test_err_file
check_test_err 3 "pu-learning"

