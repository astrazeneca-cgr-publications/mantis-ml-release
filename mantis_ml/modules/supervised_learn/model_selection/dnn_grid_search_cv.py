import pandas as pd
import numpy as np
import random
import operator
import itertools
import sys
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from multiprocessing import Process, Manager
from keras.utils import to_categorical
from keras.optimizers import Adam

from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.prepare_train_test_sets import PrepareTrainTestSets
from mantis_ml.modules.supervised_learn.classifiers.dnn import DnnClassifier




# === Default DNN parameters ===
default_dnn_params = {
    'clf_id': 'DNN',
    'regl': 0.0001,
    'hidden_layer_nodes': [128],
    'dropout_ratio': 0.1,
    'optimizer': 'Adam', #Adam(lr=0.0001),
    'epochs': 50,
    'batch_size': 64,
    'add_dropout': True,
    'verbose': False,
    'make_plots': False
}

param_val_delim = '|'
layers_size_delim = ';'




def get_consistent_balanced_datasets(stratified_kfold=True):
    total_subsets = None
    data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')
    # get random partition of the entire dataset
    iter_random_state = random.randint(0, 1000000000)

    set_generator = PrepareTrainTestSets(cfg)
    train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(data, random_state=iter_random_state,
                                                       stratified_kfold=stratified_kfold)
    if total_subsets is None:
        total_subsets = len(train_dfs)

    # Select random groups of balanced sets stratified into k-folds
    random_kfold_indexes = random.sample(range(int(total_subsets / cfg.kfold)), iterations)
    # print(random_kfold_indexes)

    train_test_indexes = []
    for i in random_kfold_indexes:
        cur_ind_list = []
        for k in range(cfg.kfold):
            cur_ind_list.append(i * cfg.kfold + k)
        # print(cur_ind_list)
        train_test_indexes.extend(cur_ind_list)
    # print(train_test_indexes)

    return train_dfs, test_dfs, train_test_indexes



def optimise_for_parameters(params_to_tune):
    '''
    Optimise for max. 2 parameters simultaneously
    :param params_to_tune: 
    :return: 
    '''
    process_jobs = []
    tmp_dnn_params = default_dnn_params.copy()

    param_combinations = parameter_space[params_to_tune[0]]
    if len(params_to_tune) == 2:
        param_combinations = list(itertools.product(parameter_space[params_to_tune[0]], parameter_space[params_to_tune[1]]))
    print(param_combinations)


    for tmp_comb in param_combinations:
        print(f"Current param. values for {params_to_tune}: {tmp_comb}")
        if len(params_to_tune) == 1:
            tmp_dnn_params[params_to_tune[0]] = tmp_comb
        else:
            for i in range(len(params_to_tune)):
                tmp_dnn_params[params_to_tune[i]] = tmp_comb[i]

        p = Process(target=run_kfold_dnn_model, args=(train_test_indexes, params_to_tune, tmp_dnn_params))
        process_jobs.append(p)
        p.start()

        if len(process_jobs) > max_workers:
            for p in process_jobs:
                p.join()
            process_jobs = []

    for p in process_jobs:
        p.join()


def run_kfold_dnn_model(train_test_indexes, params_to_tune, tmp_dnn_params):

    tmp_auc_sum = 0
    batch_cnt = 1
    fold_cnt = 1
    for i in train_test_indexes:

        train_data = train_dfs[i].copy()
        test_data = test_dfs[i].copy()

        set_generator = PrepareTrainTestSets(cfg)
        X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data,
                                                                                                        test_data)

        # convert target variables to 2D arrays
        y_train = np.array(to_categorical(y_train, 2))
        y_test = np.array(to_categorical(y_test, 2))

        dnn_model = DnnClassifier(cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names,
                                  **tmp_dnn_params)

        dnn_model.run()

        print(f"AUC (batch {batch_cnt}, fold {fold_cnt}:", dnn_model.auc_score)
        tmp_auc_sum += dnn_model.auc_score

        if fold_cnt % cfg.kfold == 0:
            fold_cnt = 0
            batch_cnt += 1
        fold_cnt += 1

    cur_avg_auc = tmp_auc_sum / cfg.kfold


    cur_param_value = ''

    if len(params_to_tune) == 1:
        cur_param = params_to_tune[0]
        cur_param_value = tmp_dnn_params[cur_param]

        if cur_param == 'hidden_layer_nodes':
            cur_param_value = layers_size_delim.join([str(i) for i in cur_param_value])
    else:
        for p in params_to_tune:
            tmp_param_value = tmp_dnn_params[p]
            if p == 'hidden_layer_nodes':
                tmp_param_value = layers_size_delim.join([str(i) for i in tmp_param_value])

            cur_param_value += str(tmp_param_value) + param_val_delim
        cur_param_value = cur_param_value[:-1]

    gridsearch_results[cur_param_value] = cur_avg_auc
    print(f"Average AUC: {cur_avg_auc}")



def update_optimal_val_for_cur_param(gridsearch_results, params_to_tune):

    global default_dnn_params

    optimal_comb = max(gridsearch_results.items(), key=operator.itemgetter(1))[0]
    print(optimal_comb)

    optimal_vals = optimal_comb.split(param_val_delim)


    for p in range(len(params_to_tune)):
        param = params_to_tune[p]
        optimal_val = optimal_vals[p]

        if param == 'hidden_layer_nodes':
            optimal_val = [int(s) for s in optimal_val.split(layers_size_delim)] # unfold hidden layer architecture in case there are > 1 hidden layers

        if param in ['dropout_ratio', 'regl']:
            default_dnn_params[param] = float(optimal_val)
        elif param in ['batch_size', 'epochs']:
            default_dnn_params[param] = int(optimal_val)
        else:
            default_dnn_params[param] = optimal_val

        print(f"Best {param}:", default_dnn_params[param])


def wrap_optimisation_call(params_to_tune):

    optimise_for_parameters(params_to_tune)
    print(f"\nAUC for {params_to_tune}:", gridsearch_results)

    update_optimal_val_for_cur_param(gridsearch_results, params_to_tune)
    print(default_dnn_params)



if __name__ == '__main__':

    config_file = sys.argv[1]
    # config_file = Path('../../../config.yaml')
    cfg = Config(config_file)

    iterations = 10
    max_workers = cfg.kfold * iterations

    manager = Manager()
    gridsearch_results = manager.dict()

    print('Retrieving random balanced datasets with Stratified k-fold... (for use with all classifiers except for Stacking')
    train_dfs, test_dfs, train_test_indexes = get_consistent_balanced_datasets(stratified_kfold=True)



    parameter_space = {'hidden_layer_nodes': [[32], [64], [128], [256], [32, 32], [64, 64], [128, 128], [64, 64, 64]],
                       'dropout_ratio': [0.1, 0.2, 0.3],
                       'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
                       'regl': [0.0001, 0.001, 0.01, 0.1],
                       'batch_size': [16, 32, 64, 128],
                       'epochs': [10, 20, 30, 50, 70, 100]
    }



    # ---------------- GridSearchCV ----------------
    params_to_tune = ['dropout_ratio', 'hidden_layer_nodes']
    wrap_optimisation_call(params_to_tune)

    params_to_tune = ['batch_size', 'epochs']
    wrap_optimisation_call(params_to_tune)

    params_to_tune = ['optimizer', 'regl']
    wrap_optimisation_call(params_to_tune)
