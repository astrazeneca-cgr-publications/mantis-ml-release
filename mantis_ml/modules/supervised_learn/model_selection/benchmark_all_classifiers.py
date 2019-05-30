import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import auc
import pandas as pd
import numpy as np
import random
import sys
import os
from pathlib import Path
import pickle  
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from palettable.colorbrewer.sequential import Greens_9

import keras
from keras.optimizers import Adam
import xgboost as xgb
from multiprocessing import Process, Manager, Pool, freeze_support

from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.prepare_train_test_sets import PrepareTrainTestSets
from mantis_ml.modules.supervised_learn.classifiers.dnn import DnnClassifier
from mantis_ml.modules.supervised_learn.classifiers.ensemble_lib import ensemble_clf_params
from mantis_ml.modules.supervised_learn.classifiers.ensemble_stacking import EnsembleClassifier
from mantis_ml.modules.supervised_learn.classifiers.sklearn_extended_classifier import SklearnExtendedClassifier

sklearn_extended_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'SVC', 'XGBoost']
feature_imp_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'XGBoost'] #for Stacking: redundant

def train_dnn_on_subset(train_data, test_data, train_acc_list, test_acc_list, auc_score_list, fpr_tpr_lists):

        set_generator = PrepareTrainTestSets(cfg)

        X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data, test_data)

        # Select Boruta confirmed features, if specified in config.yaml
        if cfg.feature_selection == 'boruta':
            important_features = pd.read_csv(str(cfg.boruta_tables_dir / 'Confirmed.boruta_features.csv'), header=None)
            important_features = important_features.iloc[:, 0].values
            
            important_feature_indexes = sorted([list(train_data.columns.values).index(f) for f in important_features])
            X_train = X_train[:, important_feature_indexes]
            X_test = X_test[:, important_feature_indexes]


        # convert target variables to 2D arrays
        y_train = np.array(keras.utils.to_categorical(y_train, 2))
        y_test = np.array(keras.utils.to_categorical(y_test, 2))


        # === DNN parameters ===
        dnn_params = ensemble_clf_params['DNN']
        dnn_params['clf_id'] = 'DNN'
        dnn_params['verbose'] = False
        dnn_params['make_plots'] = False
        # ======================

        
        dnn_model = DnnClassifier(cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, **dnn_params)

        dnn_model.run()
        print(dnn_model.train_acc, dnn_model.test_acc, dnn_model.auc_score)

        fpr, tpr = dnn_model.plot_roc_curve()
        fpr_tpr_lists.append([fpr, tpr])

        train_acc_list.append(dnn_model.train_acc)
        test_acc_list.append(dnn_model.test_acc)
        auc_score_list.append(dnn_model.auc_score)



def train_stacking_on_subset(selected_base_classifiers, final_level_classifier,
                             train_data, test_data, feature_dataframe_list, 
                             train_acc_list, test_acc_list, auc_score_list, fpr_tpr_lists):

    set_generator = PrepareTrainTestSets(cfg)

    X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data, test_data)
        
    # Select Boruta confirmed features, if specified in config.yaml
    if cfg.feature_selection == 'boruta':
        important_features = pd.read_csv(str(cfg.boruta_tables_dir / 'Confirmed.boruta_features.csv'), header=None)
        important_features = important_features.iloc[:, 0].values
        print('Important features: ', len(important_features))
            
        important_feature_indexes = sorted([list(train_data.columns.values).index(f) for f in important_features])
        X_train = X_train[:, important_feature_indexes]
        X_test = X_test[:, important_feature_indexes]


    base_sklearn_clf_params = dict((k, v) for k, v in ensemble_clf_params.items() if k in selected_base_classifiers)
    final_clf_params = {final_level_classifier: ensemble_clf_params[final_level_classifier]}

    # Create Stacking classifier instance
    stack_clf = EnsembleClassifier(cfg, base_sklearn_clf_params, final_clf_params)
    stack_clf.build_base_classifiers()

    # gEt Stacking predictions
    stack_clf.get_base_level_predictions(X_train, y_train, X_test)

    # concatenate first level predictions to feed to the final layer
    final_lvl_X_train, final_lvl_X_test = stack_clf.concat_first_lvl_predictions()


    # ------------------------------------------------------------------------------------------------------
    if cfg.add_original_features_in_stacking:
        final_lvl_X_train = np.concatenate((X_train, final_lvl_X_train), axis=1)
        final_lvl_X_test = np.concatenate((X_test, final_lvl_X_test), axis=1)
    # ------------------------------------------------------------------------------------------------------


    stack_clf.run_final_level_classifier(final_lvl_X_train, y_train, final_lvl_X_test, y_test, train_gene_names,
                                         test_gene_names)


    fpr, tpr = stack_clf.final_model.plot_roc_curve()
    fpr_tpr_lists.append([fpr, tpr])
    
    train_acc_list.append(stack_clf.train_acc)
    test_acc_list.append(stack_clf.test_acc)
    auc_score_list.append(stack_clf.auc_score)



def train_extended_sklearn_clf_on_subset(clf_id, train_data, test_data, feature_dataframe_list, 
                                         train_acc_list, test_acc_list, auc_score_list,
                                         fpr_tpr_lists):

    set_generator = PrepareTrainTestSets(cfg)

    X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data, test_data)


    # Select Boruta confirmed features, if specified in config.yaml
    important_feature_indexes = list(range(X_train.shape[1]))
    if cfg.feature_selection == 'boruta':
        important_features = pd.read_csv(str(cfg.boruta_tables_dir / 'Confirmed.boruta_features.csv'), header=None)
        important_features = important_features.iloc[:, 0].values
            
        important_feature_indexes = sorted([list(train_data.columns.values).index(f) for f in important_features])
        X_train = X_train[:, important_feature_indexes]
        X_test = X_test[:, important_feature_indexes]


    make_plots = False
    verbose = False
    # Possible clf ids: 'RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'SVC', 'XGBoost'

    clf_params = ensemble_clf_params[clf_id]

    model = SklearnExtendedClassifier(cfg, X_train, y_train, X_test, y_test,
                                      train_gene_names, test_gene_names, clf_id, clf_params, make_plots, verbose)

    if clf_id == 'XGBoost':
        model.model = xgb.XGBClassifier(**clf_params)
    else:
        model.build_model()

    model.train_model()
    
    # get feature importance
    #print('important_feature_indexes:', important_feature_indexes)
    feature_cols = train_data.iloc[:, important_feature_indexes].columns.values
    if clf_id == 'XGBoost':
        cur_model = model.model
        cur_feature_imp = cur_model.feature_importances_ 
        tmp_feature_dataframe = pd.DataFrame({'features': feature_cols, clf_id: cur_feature_imp})
        feature_dataframe_list.append(tmp_feature_dataframe)
    elif clf_id in feature_imp_classifiers:
        model.get_feature_importance(X_train, y_train, feature_cols, clf_id)
        feature_dataframe_list.append(model.feature_dataframe)

    model.process_predictions()
    fpr, tpr = model.plot_roc_curve()
    fpr_tpr_lists.append([fpr, tpr])

    train_acc_list.append(model.train_acc)
    test_acc_list.append(model.test_acc)
    auc_score_list.append(model.auc_score)



def get_aggregate_feature_importance(clf_id, feature_dataframe_list):

    if clf_id in feature_imp_classifiers:
        feat_list_file = str(cfg.benchmark_out / ('PU_' + clf_id + '.feature_dfs_list.txt'))
        # cleanup
        if os.path.exists(feat_list_file):
            os.remove(feat_list_file)
        with open(feat_list_file, 'a') as f:     
            for df in feature_dataframe_list:         
                del df['features']
                df.to_csv(f)

        # Aggregate feature importance
        avg_feature_dataframe = None
        for feature_df in feature_dataframe_list:
            feature_df.index = feature_df['features']
            del feature_df['features']

            if avg_feature_dataframe is None:
                avg_feature_dataframe = feature_df
            else:
                avg_feature_dataframe.add(feature_df, fill_value=0)

        avg_feature_dataframe = avg_feature_dataframe/len(feature_dataframe_list)
        avg_feature_dataframe.to_csv(cfg.benchmark_out / ('PU_'+clf_id+'.avg_feature_importance.tsv'), sep='\t')



def run_pu_learning(train_dfs, test_dfs, train_test_indexes, clf_id, max_workers, iterations, selected_base_classifiers=None, final_level_classifier=None):

    manager = Manager()

    feature_dataframe_list = manager.list()
    train_acc_list = manager.list()
    test_acc_list = manager.list()
    auc_score_list = manager.list()

    fpr_tpr_lists = manager.list()
    #tpr_lists = manager.list()


    process_jobs = []


    for i in train_test_indexes:
        train_data = train_dfs[i]
        test_data = test_dfs[i]
        #print(f"Training set size: {train_data.shape[0]}")
        #print(f"Test set size: {test_data.shape[0]}")

        p = None
        if clf_id == 'DNN':
            p = Process(target=train_dnn_on_subset, args=(train_data, test_data,
                                                          train_acc_list, test_acc_list, 
                                                          auc_score_list, fpr_tpr_lists))
        elif clf_id.startswith('Stacking'):
            # Define Stacking classifiers
            p = Process(target=train_stacking_on_subset, args=(selected_base_classifiers, final_level_classifier,
                                                               train_data, test_data, feature_dataframe_list, 
                                                               train_acc_list, test_acc_list, auc_score_list, fpr_tpr_lists))
        elif clf_id in sklearn_extended_classifiers:
            p = Process(target=train_extended_sklearn_clf_on_subset, args=(clf_id,
                                                              train_data, test_data,
                                                              feature_dataframe_list, train_acc_list,
                                                              test_acc_list, auc_score_list,
                                                              fpr_tpr_lists))

        process_jobs.append(p)
        p.start()
        
        
        if len(process_jobs) > max_workers:
           for p in process_jobs:
               p.join()
           process_jobs = []


    for p in process_jobs:
        p.join()


    get_aggregate_feature_importance(clf_id, feature_dataframe_list)


    all_roc_data = pd.DataFrame()
    for i in range(len(fpr_tpr_lists)):
        fpr = fpr_tpr_lists[i][0]
        tpr = fpr_tpr_lists[i][1]

        tmp_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr}, index=range(len(fpr)))
        if len(all_roc_data) > 0:
            all_roc_data = pd.concat([all_roc_data, tmp_df], axis=0)
        else:
            all_roc_data = tmp_df
    all_roc_data.sort_values(by=['TPR'], inplace=True)

    return auc_score_list, fpr_tpr_lists, all_roc_data




def plot_aggregate_roc_curves(all_clf, fpr_tpr_dict, auc_scores_dict, overlap_plots=True):

    if overlap_plots:
        f = plt.figure(figsize=(6, 6))

  
    avg_auc_distr_per_clf = pd.DataFrame()

    for clf_id in all_clf:        
        print('> ' + clf_id)
        if not overlap_plots:
            f = plt.figure(figsize=(6, 6))

        fpr_tpr_lists = fpr_tpr_dict[clf_id]
        auc_score_list = auc_scores_dict[clf_id]
        
        avg_auc_score = round(sum(auc_score_list) / len(auc_score_list), 3)
        print('AUC:', avg_auc_score) 

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        batch_cnt = 1
        fold_cnt = 1
        cur_avg_auc = 0
        avg_aucs_per_fold = list()
        for i in range(len(fpr_tpr_lists)):
            fpr = fpr_tpr_lists[i][0]
            tpr = fpr_tpr_lists[i][1]
            auc_score = auc_score_list[i]
            cur_avg_auc += auc_score

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(auc_score)

            if not overlap_plots: 
                if fold_cnt % cfg.kfold == 0:
                    #add label for every batch of k-fold
                    cur_avg_auc /= cfg.kfold
                    avg_aucs_per_fold.append(cur_avg_auc)
                    plt.plot(fpr, tpr, lw=0.5, alpha=0.3, label="ROC: batch %s, (Avg. AUC = %0.3f)" % (batch_cnt, cur_avg_auc), color=clf_colors[clf_id])
                    cur_avg_auc = 0
                else:
                    plt.plot(fpr, tpr, lw=0.5, alpha=0.3, color=clf_colors[clf_id])

                #plt.plot(fpr, tpr, lw=0.5, alpha=0.3, label='ROC: batch %d, fold %d (AUC = %0.3f)' % (batch_cnt, fold_cnt, auc_score), color=clf_colors[clf_id])

            if fold_cnt % cfg.kfold == 0:
                fold_cnt = 0
                batch_cnt += 1
            fold_cnt += 1
    
        avg_auc_distr_per_clf[clf_id] = avg_aucs_per_fold


        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=clf_colors[clf_id], 
             label=r'%s (AUC = %0.3f $\pm$ %0.3f)' % (clf_id, mean_auc, std_auc),
             lw=1, alpha=.8)


        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right", fontsize='x-small', bbox_to_anchor=(1.5, 0))

        if not overlap_plots:
            std_tpr = np.std(tprs, axis=0) 
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0) 
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', 
                             alpha=.2, label=r'$\pm$ 1 std. dev.')

            plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance', alpha=.8, linewidth=1)
            f.savefig(str(cfg.benchmark_out / (clf_id + ".ROC_curve.pdf")), bbox_inches='tight')
    

    if overlap_plots:
        plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance', alpha=.8, linewidth=0.7)
        f.savefig(str(cfg.benchmark_out / "All_classifiers.ROC_curve.pdf"), bbox_inches='tight')


    if not overlap_plots:
        print('>>> AVG AUC DISTR DF:')
        print(avg_auc_distr_per_clf)

        avg_auc_distr_per_clf = avg_auc_distr_per_clf.reindex(avg_auc_distr_per_clf.mean().sort_values().index, axis=1)
        print(avg_auc_distr_per_clf)

        # Plot Boxplot
        boxplot_pallete = Greens_9.hex_colors + ['#525252', '#252525', '#000000']

        fig, ax = plt.subplots(figsize=(16, 8))
        position = 0
        xtick_labels = []
        for clf_id in avg_auc_distr_per_clf.columns.values:

            cur_clf_series = avg_auc_distr_per_clf[clf_id]

            bp = ax.boxplot(cur_clf_series, positions=[position], patch_artist=True, notch=False, widths=0.4,
                            flierprops=dict(marker='o', markerfacecolor='black', markersize=3, linestyle='dotted'))

            xtick_labels.append(clf_id + '\n(Avg. AUC: ' + str(round(cur_clf_series.mean(), 3)) + ')')
            cur_face_color = boxplot_pallete[position]

            for patch in bp['boxes']:
                _ = patch.set(facecolor=cur_face_color, alpha=0.9)

            position += 1


        _ = ax.set_title('[Benchmarking] AUC scores from all classifiers in: ' + cfg.phenotype, fontsize=20)
        _ = ax.set_xticks(range(position + 1))
        _ = ax.set_xticklabels(xtick_labels, rotation=90)
        _ = ax.set_xlim(xmin=-0.5)
        #_ = ax.set_ylim([0.74, 0.84])
        _ = ax.set_xlabel('Classifiers', fontsize=18)
        _ = ax.set_ylabel('AUC score distribution across all runs', fontsize=18)
        # plt.show()

        fig.savefig(str(cfg.benchmark_out / 'Benchmarking.All_classifiers.AUC_distr_boxplots.pdf'), bbox_inches='tight')



def get_consistent_balanced_datasets(stratified_kfold=True):

    total_subsets = None
    data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')
    # get random partition of the entire dataset
    iter_random_state = random.randint(0, 1000000000)

    set_generator = PrepareTrainTestSets(cfg)
    train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(data, random_state=iter_random_state, stratified_kfold=stratified_kfold)
    if total_subsets is None:
        total_subsets = len(train_dfs)

    print('Len train_dfs:', len(train_dfs))
    print('total_subsets:', total_subsets)
    print('kfold:', cfg.kfold)

    # Select random groups of balanced sets stratified into k-folds
    random_kfold_indexes = random.sample(range(int(total_subsets/cfg.kfold)), iterations) 
    #print(random_kfold_indexes) 
    train_test_indexes = []
    for i in random_kfold_indexes:
        cur_ind_list = []
        for k in range(cfg.kfold):
            cur_ind_list.append(i * cfg.kfold + k)
        #print(cur_ind_list)
        train_test_indexes.extend(cur_ind_list)
    print(train_test_indexes)
 
    return train_dfs, test_dfs, train_test_indexes


def store_reproducible_partitioning_lists():
   
    print('Retrieving random balanced datasets with Stratified k-fold... (for use with all classifiers except for Stacking)')
    # ----  Create random balanced datasets with Stratified k-fold to be used by all classifiers except 'Stacking' ----
    train_dfs, test_dfs, train_test_indexes = get_consistent_balanced_datasets(stratified_kfold=True)
    # -----------------------------------------------------------------------------------------------------------------

    print('Retrieving random balanced datasets for Stacking without Stratified k-fold...(for use with Stacking classifiers)')
    # ----  Create random balanced datasets without stratified k-fold to be used by Stacking only ----
    stacking_train_dfs, stacking_test_dfs, stacking_train_test_indexes = get_consistent_balanced_datasets(stratified_kfold=False)
    # ------------------------------------------------------------------------------------------------

    if not os.path.exists(static_partition_dir):
        os.makedirs(static_partition_dir)

    lists_to_store = ['train_dfs', 'test_dfs', 'train_test_indexes', 'stacking_train_dfs', 'stacking_test_dfs', 'stacking_train_test_indexes'] 

    for l in lists_to_store:
        with open((static_partition_dir + '/' + l + '.pkl'), 'wb') as output:
            pickle.dump(eval(l), output, pickle.HIGHEST_PROTOCOL)


def read_reproducible_partitioning_lists():
    
    lists_to_store = ['train_dfs', 'test_dfs', 'train_test_indexes', 'stacking_train_dfs', 'stacking_test_dfs', 'stacking_train_test_indexes'] 

    results = []

    for l in lists_to_store:
        print('>', l + ':')
        with open((static_partition_dir + '/' + l + '.pkl'), 'rb') as input:
            tmp_list = pickle.load(input)
            print(len(tmp_list), '\n')
            results.append(tmp_list)

    return results


if __name__ == '__main__':

    config_file = sys.argv[1]
    iterations = int(sys.argv[2]) #10 

    # Total datasets to run for benchmarking:
    # cfg.kfold * iterations -- typically 100

    # config_file = Path('../../../config.yaml')
    cfg = Config(config_file)


    # ======= Parameters for Benchmarking =======
    create_reproducible_train_test_sets = True
    read_reproducible_train_test_sets = True

    max_workers = 10 #100 #cfg.kfold * iterations
   # ============================================


    static_partition_dir = 'reproducible_partition_pickle_files'
    
    auc_scores_dict = {}
    fpr_tpr_dict = {}
    all_roc_data_dict = {}

    all_clf = ['Stacking_DNN', 'DNN', 'ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'XGBoost']
    #all_clf = ['DNN', 'ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'XGBoost']
    clf_colors = {'ExtraTreesClassifier': '#33a02c', 'RandomForestClassifier': '#1f78b4', 
                  'GradientBoostingClassifier': '#e31a1c', 'SVC': '#6a3d9a', 'XGBoost': '#ff7f00',
                  'DNN': '#67001f', 'Stacking_DNN': '#252525', 'Stacking_XGBoost': '#ae017e'}


    
    if create_reproducible_train_test_sets:
        store_reproducible_partitioning_lists()
        #sys.exit('Stored reproducible train/test sets into pikcle files under ' + static_partition_dir)

    if read_reproducible_train_test_sets:
        train_dfs, test_dfs, train_test_indexes, stacking_train_dfs, stacking_test_dfs, stacking_train_test_indexes = read_reproducible_partitioning_lists()
        print('Read reproducible train/test sets from pikcle files under ' + static_partition_dir)

        print(len(train_test_indexes))

    else:
        print('Retrieving random balanced datasets with Stratified k-fold... (for use with all classifiers except for Stacking)')
        # ----  Create random balanced datasets with Stratified k-fold to be used by all classifiers except 'Stacking' ----
        train_dfs, test_dfs, train_test_indexes = get_consistent_balanced_datasets(stratified_kfold=True)
        # -----------------------------------------------------------------------------------------------------------------

        print('Retrieving random balanced datasets for Stacking without Stratified k-fold...(for use with Stacking classifiers)')
        # ----  Create random balanced datasets without stratified k-fold to be used by Stacking only ----
        stacking_train_dfs, stacking_test_dfs, stacking_train_test_indexes = get_consistent_balanced_datasets(stratified_kfold=False)
        # ------------------------------------------------------------------------------------------------



    for clf_id in all_clf:

        auc_score_list, fpr_tpr_lists, all_roc_data = None, None, None
   
        if clf_id.startswith('Stacking'):
            final_level_classifier = clf_id.replace('Stacking_', '')
            selected_base_classifiers = ['ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC']

            train_dfs, test_dfs, train_test_indexes = stacking_train_dfs, stacking_test_dfs, stacking_train_test_indexes

            auc_score_list, fpr_tpr_lists, all_roc_data = run_pu_learning(train_dfs, test_dfs, train_test_indexes, 
                                                                          clf_id, max_workers, iterations,
                                                                          selected_base_classifiers=selected_base_classifiers,
                                                                          final_level_classifier=final_level_classifier)
        else:
            auc_score_list, fpr_tpr_lists, all_roc_data = run_pu_learning(train_dfs, test_dfs, train_test_indexes, clf_id, max_workers, iterations)
    
        auc_scores_dict[clf_id] = auc_score_list
        fpr_tpr_dict[clf_id] = fpr_tpr_lists
        all_roc_data_dict[clf_id] = all_roc_data            


    # - overlap_plots = True: plot smooth ROC curves from all classifiers in same plot
    #                   False: plot ROC curves for each fold and smooth ROC curve for each classifier in separate file
    plot_aggregate_roc_curves(all_clf, fpr_tpr_dict, auc_scores_dict, overlap_plots=True) 
    plot_aggregate_roc_curves(all_clf, fpr_tpr_dict, auc_scores_dict, overlap_plots=False)
