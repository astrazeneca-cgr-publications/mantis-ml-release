import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import glob, os, sys
import random
from multiprocessing import Process, Manager, Pool, freeze_support
import subprocess

from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.prepare_train_test_sets import PrepareTrainTestSets
from  mantis_ml.modules.supervised_learn.feature_selection.boruta_score_aggregator import BorutaScoreAggregator


def call_boruta_r_wrapper(cfg, train_data_file, test_data_file, tmp_run_id, verbose=1):

    config_base_path = cfg.config_dir_path

    cmd = ' '.join(['Rscript', config_base_path + '/modules/supervised_learn/feature_selection/boruta_feature_selection.R "', str(train_data_file), '" "', str(test_data_file), '" "',
                    str(cfg.boruta_figs_dir), '" "',
                    str(cfg.boruta_tables_dir), '" "', str(tmp_run_id), '"'])

    print(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #for line in p.stdout:
    for line in p.stderr:
        print(line)
    p.stdout.close()
    p.wait()

    #p.wait()
    # stdout, stderr = p.communicate()
    #
    # if verbose:
    #     stdout = str(stdout, "utf-8")
    #     stderr = str(stderr, "utf-8")
    #     print('>', tmp_run_id, 'STDOUT:\n\n', stdout)
    #     print('>', tmp_run_id, 'STDERR:\n\n', stderr)


class BorutaWrapper:

    def __init__(self, cfg):
        self.cfg = cfg


    def run(self, boruta_iterations=None):

        if boruta_iterations is None:
            boruta_iterations = self.cfg.boruta_iterations

        print("\n>> Running Boruta feature selection algorithm...")
        set_generator = PrepareTrainTestSets(self.cfg)

        data = pd.read_csv(self.cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')

        # create tmp dir for intermediate results
        tmp_dir = self.cfg.boruta_tables_dir / 'tmp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)
        # create out dir with boruta results
        out_dir = self.cfg.boruta_tables_dir / 'out'
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)

        # === Boruta run paramters ===
        max_workers = self.cfg.nthreads
        # ===========================

        total_runs = 0
        process_jobs = []

        i = 0
        while True:
            i += 1 
            print('-----------------------------------------------> Iteration:', i)
            process_jobs = []

            # get random partition of the entire dataset
            iter_random_state = random.randint(0, 1000000000)
            train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(data, random_state=iter_random_state)

            # Loop through all balanced datasets from the entire partitioning of current iteration
            for i in range(len(train_dfs)):
                print('\n>>> Total runs (so far):', total_runs)

                train_data = train_dfs[i]
                test_data = test_dfs[i]

                tmp_run_id = str(iter_random_state) + '_' + str(i)

                tmp_train_file = tmp_dir / ('train.' + tmp_run_id + '.txt')
                tmp_test_file = tmp_dir / ('test.' + tmp_run_id + '.txt')
                print(tmp_train_file)
                print(tmp_test_file)

                train_data.to_csv(tmp_train_file, index=False)
                test_data.to_csv(tmp_test_file, index=False)

                p = Process(target=call_boruta_r_wrapper, args=(self.cfg, tmp_train_file, tmp_test_file, tmp_run_id))

                process_jobs.append(p)
                p.start()

                if len(process_jobs) > max_workers:
                    for p in process_jobs:
                        p.join()
                    process_jobs = []

                total_runs += 1
                if (total_runs % max_workers == 0) or (total_runs >= boruta_iterations):
                    break

            if total_runs >= boruta_iterations:
                break

        for p in process_jobs:
            p.join()

        agg = BorutaScoreAggregator(self.cfg, str(self.cfg.boruta_tables_dir / 'out'))
        agg.get_imp_dfs()
        agg.get_final_decisions()
        agg.get_boruta_boxplots()



if __name__ == '__main__':

    #config_file = '../../../config.yaml'
    config_file = sys.argv[1]
    cfg = Config(config_file)

    boru_wrapper = BorutaWrapper(cfg)
    boru_wrapper.run()
