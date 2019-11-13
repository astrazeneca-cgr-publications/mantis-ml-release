import pandas as pd
import numpy as np
import random
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import xgboost as xgb
from multiprocessing import Process, Manager

from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.prepare_train_test_sets import PrepareTrainTestSets
from mantis_ml.modules.supervised_learn.classifiers.dnn import DnnClassifier
from mantis_ml.modules.supervised_learn.classifiers.ensemble_lib import ensemble_clf_params
from mantis_ml.modules.supervised_learn.classifiers.ensemble_stacking import EnsembleClassifier
from mantis_ml.modules.supervised_learn.classifiers.sklearn_extended_classifier import SklearnExtendedClassifier
import keras
from keras.optimizers import Adam

sklearn_extended_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'SVC', 'XGBoost']
feature_imp_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'XGBoost']



class PULearning:

    def __init__(self, cfg, data, clf_id, final_level_classifier=None):
        self.cfg = cfg
        self.data = data
        self.clf_id = clf_id
        self.final_level_classifier = final_level_classifier

        if self.clf_id == 'Stacking' and self.final_level_classifier is None:
                sys.exit('\n[Error -- pu_learning.py]: Please define final level classifier for Stacking classifier')


    def train_dnn_on_subset(self, train_data, test_data, pos_genes_dict, neg_genes_dict, train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict):

            set_generator = PrepareTrainTestSets(self.cfg)
            X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data, test_data)
    
            # Select Boruta confirmed features, if specified in config.yaml
            if self.cfg.feature_selection == 'boruta':
                try:
                    important_features = pd.read_csv(str(self.cfg.boruta_tables_dir / 'Confirmed.boruta_features.csv'), header=None)
                    important_features = important_features.iloc[:, 0].values
    
                    important_feature_indexes = sorted([list(train_data.columns.values).index(f) for f in important_features])
                    X_train = X_train[:, important_feature_indexes]
                    X_test = X_test[:, important_feature_indexes]
                except Exception as e:
                    print(str() + '\nCoud not find file: Confirmed.boruta_features.csv')
    
            # convert target variables to 2D arrays
            y_train = np.array(keras.utils.to_categorical(y_train, 2))
            y_test = np.array(keras.utils.to_categorical(y_test, 2))
    
            # === DNN parameters ===
            dnn_params = ensemble_clf_params['DNN']
            dnn_params['clf_id'] = 'DNN'
            dnn_params['verbose'] = False
            dnn_params['make_plots'] = False
            # ======================
    
            dnn_model = DnnClassifier(self.cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, **dnn_params)
    
            dnn_model.run()
            print(dnn_model.train_acc, dnn_model.test_acc, dnn_model.auc_score)
    
    
            positive_y_prob = dnn_model.y_prob[:, 1]
    
            for g in range(len(test_gene_names)):
                cur_gene = test_gene_names.values[g]
    
                if cur_gene not in pos_y_prob_dict:
                    pos_y_prob_dict[cur_gene] = [positive_y_prob[g]]
                else:
                    pos_y_prob_dict[cur_gene] = pos_y_prob_dict[cur_gene] + [positive_y_prob[g]]
    
    
            train_acc_list.append(dnn_model.train_acc)
            test_acc_list.append(dnn_model.test_acc)
            auc_score_list.append(dnn_model.auc_score)
    
            tmp_pos_genes = list(dnn_model.tp_genes.values)
            tmp_pos_genes.extend(dnn_model.fp_genes.values)
    
            tmp_neg_genes = list(dnn_model.tn_genes.values)
            tmp_neg_genes.extend(dnn_model.fn_genes.values)
    
            for p in sorted(tmp_pos_genes):
                pos_genes_dict[p] = pos_genes_dict.get(p, 0) + 1
    
            for n in sorted(tmp_neg_genes):
                neg_genes_dict[n] = neg_genes_dict.get(n, 0) + 1
    
    
    
    def train_stacking_on_subset(self, selected_base_classifiers, final_level_classifier,
                                 train_data, test_data, pos_genes_dict, neg_genes_dict,
                                 feature_dataframe_list, train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict):

        set_generator = PrepareTrainTestSets(self.cfg)
        X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data, test_data)
    
        # Select Boruta confirmed features, if specified in config.yaml
        if self.cfg.feature_selection == 'boruta':
            try:
                important_features = pd.read_csv(str(self.cfg.boruta_tables_dir / 'Confirmed.boruta_features.csv'), header=None)
                important_features = important_features.iloc[:, 0].values
                print('Important features: ', len(important_features))
    
                important_feature_indexes = sorted([list(train_data.columns.values).index(f) for f in important_features])
                X_train = X_train[:, important_feature_indexes]
                X_test = X_test[:, important_feature_indexes]
            except Exception as e:
                print(str() + '\nCoud not find file: Confirmed.boruta_features.csv')
    
        base_sklearn_clf_params = dict((k, v) for k, v in ensemble_clf_params.items() if k in selected_base_classifiers)
        final_clf_params = {final_level_classifier: ensemble_clf_params[final_level_classifier]}
    
        # Create Stacking classifier instance
        stack_clf = EnsembleClassifier(self.cfg, base_sklearn_clf_params, final_clf_params)
    
        stack_clf.build_base_classifiers()
    
        # Extract feature importance
        #feature_cols = train_data.drop([self.cfg.Y], axis=1).columns.values
        #stack_clf.get_feature_imp_per_base_clf(X_train, y_train, feature_cols)
        #feature_dataframe_list.append(stack_clf.feature_dataframe)
    
        # Get Stacking predictions
        stack_clf.get_base_level_predictions(X_train, y_train, X_test)
    
        # concatenate first level predictions to feed to the final layer
        final_lvl_X_train, final_lvl_X_test = stack_clf.concat_first_lvl_predictions()
    
    
        # ------------------------------------------------------------------------------------------------------
        if self.cfg.add_original_features_in_stacking:
            final_lvl_X_train = np.concatenate((X_train, final_lvl_X_train), axis=1)
            final_lvl_X_test = np.concatenate((X_test, final_lvl_X_test), axis=1)
            # ------------------------------------------------------------------------------------------------------
    
    
        stack_clf.run_final_level_classifier(final_lvl_X_train, y_train, final_lvl_X_test, y_test, train_gene_names,
                                             test_gene_names)
    
    
        positive_y_prob = stack_clf.y_prob[:, 1]
    
        for g in range(len(test_gene_names)):
            cur_gene = test_gene_names.values[g]
    
            if cur_gene not in pos_y_prob_dict:
                pos_y_prob_dict[cur_gene] = [positive_y_prob[g]]
            else:
                pos_y_prob_dict[cur_gene] = pos_y_prob_dict[cur_gene] + [positive_y_prob[g]]
    
        
        train_acc_list.append(stack_clf.train_acc)
        test_acc_list.append(stack_clf.test_acc)
        auc_score_list.append(stack_clf.auc_score)
    
        tmp_pos_genes = list(stack_clf.tp_genes.values)
        tmp_pos_genes.extend(stack_clf.fp_genes.values)
    
        tmp_neg_genes = list(stack_clf.tn_genes.values)
        tmp_neg_genes.extend(stack_clf.fn_genes.values)
    
        for p in sorted(tmp_pos_genes):
            pos_genes_dict[p] = pos_genes_dict.get(p, 0) + 1
    
        for n in sorted(tmp_neg_genes):
            neg_genes_dict[n] = neg_genes_dict.get(n, 0) + 1
    
    
    def train_extended_sklearn_clf_on_subset(self, train_data, test_data, pos_genes_dict, neg_genes_dict,
                                             feature_dataframe_list, train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict):

        set_generator = PrepareTrainTestSets(self.cfg)
        X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data, test_data)
    
        # Select Boruta confirmed features, if specified in config.yaml
        important_feature_indexes = list(range(X_train.shape[1]))
        if self.cfg.feature_selection == 'boruta':
            try:
                important_features = pd.read_csv(str(self.cfg.boruta_tables_dir / 'Confirmed.boruta_features.csv'), header=None)
                important_features = important_features.iloc[:, 0].values
    
                important_feature_indexes = sorted([list(train_data.columns.values).index(f) for f in important_features])
                X_train = X_train[:, important_feature_indexes]
                X_test = X_test[:, important_feature_indexes]
            except Exception as e:
                print(str() + '\nCoud not find file: Confirmed.boruta_features.csv')
    
        make_plots = False
        verbose = False

        clf_params = ensemble_clf_params[self.clf_id]
    
        model = SklearnExtendedClassifier(self.cfg, X_train, y_train, X_test, y_test,
                                          train_gene_names, test_gene_names, self.clf_id, clf_params, make_plots, verbose)
    
        if self.clf_id == 'XGBoost':
            model.model = xgb.XGBClassifier(**clf_params)
        else:
            model.build_model()
    
        model.train_model()
    
        # get feature importance
        # print('important_feature_indexes:', important_feature_indexes)
        feature_cols = train_data.iloc[:, important_feature_indexes].columns.values
        if self.clf_id == 'XGBoost':
            cur_model = model.model
            cur_feature_imp = cur_model.feature_importances_ 
            tmp_feature_dataframe = pd.DataFrame({'features': feature_cols, self.clf_id: cur_feature_imp})
            feature_dataframe_list.append(tmp_feature_dataframe)
        elif self.clf_id in feature_imp_classifiers:
            model.get_feature_importance(X_train, y_train, feature_cols, self.clf_id)
            feature_dataframe_list.append(model.feature_dataframe)
    
        model.process_predictions()
    
        positive_y_prob = model.y_prob[:, 1]
    
        for g in range(len(test_gene_names)):
            cur_gene = test_gene_names.values[g]
    
            if cur_gene not in pos_y_prob_dict:
                pos_y_prob_dict[cur_gene] = [positive_y_prob[g]]
            else:
                pos_y_prob_dict[cur_gene] = pos_y_prob_dict[cur_gene] + [positive_y_prob[g]]
    
        train_acc_list.append(model.train_acc)
        test_acc_list.append(model.test_acc)
        auc_score_list.append(model.auc_score)
    
        tmp_pos_genes = list(model.tp_genes.values)
        tmp_pos_genes.extend(model.fp_genes.values)
    
        tmp_neg_genes = list(model.tn_genes.values)
        tmp_neg_genes.extend(model.fn_genes.values)
    
        for p in sorted(tmp_pos_genes):
            pos_genes_dict[p] = pos_genes_dict.get(p, 0) + 1
    
        for n in sorted(tmp_neg_genes):
            neg_genes_dict[n] = neg_genes_dict.get(n, 0) + 1
    
    
    
    def run_pu_learning(self, selected_base_classifiers=None, final_level_classifier=None):
    
        manager = Manager()
        pos_genes_dict = manager.dict()
        neg_genes_dict = manager.dict()
        pos_y_prob_dict = manager.dict()
    
        feature_dataframe_list = manager.list()
        train_acc_list = manager.list()
        test_acc_list = manager.list()
        auc_score_list = manager.list()
        process_jobs = []

        total_subsets = None
        for i in range(1, self.cfg.iterations + 1):
            print('-----------------------------------------------> Iteration:', i)
    
            process_jobs = []
    
            # get random partition of the entire dataset
            iter_random_state = random.randint(0, 1000000000)

            set_generator = PrepareTrainTestSets(self.cfg)
            train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(self.data, random_state=iter_random_state)
            if total_subsets is None:
                total_subsets = len(train_dfs)
    
    
            # Loop through all balanced datasets from the entire partitioning of current iteration
            for i in range(len(train_dfs)):
            #for i in range(10): # Debugging only
                train_data = train_dfs[i]
                test_data = test_dfs[i]
                #print(f"Training set size: {train_data.shape[0]}")
                #print(f"Test set size: {test_data.shape[0]}")
    
                p = None
                if self.clf_id == 'DNN':
                    p = Process(target=self.train_dnn_on_subset, args=(train_data, test_data,
                                                                  pos_genes_dict, neg_genes_dict,
                                                                  train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict))
                elif self.clf_id == 'Stacking':
                    # Define Stacking classifiers
                    p = Process(target=self.train_stacking_on_subset, args=(selected_base_classifiers, final_level_classifier,
                                                                       train_data, test_data,
                                                                       pos_genes_dict, neg_genes_dict,
                                                                       feature_dataframe_list, train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict))
                elif self.clf_id in sklearn_extended_classifiers:
                    p = Process(target=self.train_extended_sklearn_clf_on_subset, args=(train_data, test_data,
                                                                       pos_genes_dict, neg_genes_dict,
                                                                       feature_dataframe_list, train_acc_list,
                                                                       test_acc_list, auc_score_list, pos_y_prob_dict))
    
                process_jobs.append(p)
                p.start()

                if len(process_jobs) > self.cfg.nthreads:
                    for p in process_jobs:
                        p.join()
                    process_jobs = []
    
        for p in process_jobs:
            p.join()
    
    
        if self.clf_id in feature_imp_classifiers:
            feat_list_file = str(self.cfg.superv_out / ('PU_' + self.clf_id + '.feature_dfs_list.txt'))
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
            avg_feature_dataframe.to_csv(self.cfg.superv_out / ('PU_'+self.clf_id+'.avg_feature_importance.tsv'), sep='\t')
    
        avg_train_acc = round(sum(train_acc_list) / len(train_acc_list), 2)
        avg_test_acc = round(sum(test_acc_list) / len(test_acc_list), 2)
        avg_auc_score = round(sum(auc_score_list) / len(auc_score_list), 2)
    
        #print('\nAvg. training accuracy: ' + str(avg_train_acc) + ' %')
        #print('Avg. test accuracy: ' + str(avg_test_acc) + ' %')
        print('Avg. AUC score: ' + str(avg_auc_score))
    
    

        metrics_df = pd.DataFrame(list(zip(train_acc_list, test_acc_list, auc_score_list)), columns=['Train_Accuracy', 'Test_Accuracy', 'AUC'])
        metrics_df.to_csv(self.cfg.superv_out / ('PU_'+self.clf_id+'.evaluation_metrics.tsv'), sep='\t')
    
        neg_genes_df = pd.DataFrame(list(neg_genes_dict.values()), index=(neg_genes_dict.keys()),
                                    columns=['negative_genes'])
        neg_genes_df.sort_index(inplace=True)
        print(neg_genes_df.shape)
    
        pos_genes_df = pd.DataFrame(list(pos_genes_dict.values()), index=(pos_genes_dict.keys()),
                                    columns=['positive_genes'])
        pos_genes_df.sort_index(inplace=True)
        print(pos_genes_df.shape)
    
        all_genes_df = neg_genes_df.join(pos_genes_df, how='outer')
        all_genes_df.fillna(0, inplace=True)
    
    
        print(all_genes_df.head())
        print(all_genes_df.loc[all_genes_df.index.intersection(['PKD1', 'PKD2', 'NOTCH1', 'ADIPOQ'])])
        print(all_genes_df.shape)
        all_genes_df.to_csv(self.cfg.superv_out / ('PU_'+self.clf_id+'.all_genes_predictions.tsv'), sep='\t')
    
      
    
        gene_proba_dict = pos_y_prob_dict.copy()
        gene_proba_df = pd.DataFrame.from_dict(gene_proba_dict, orient='index').T         
        gene_proba_df = gene_proba_df.reindex(gene_proba_df.mean().sort_values(ascending=False).index, axis=1)
    
        gene_proba_df.to_hdf(self.cfg.superv_proba_pred / (self.clf_id + '.all_genes.predicted_proba.h5'), key='df', mode='w')
        

    def run(self):
        if self.clf_id == 'Stacking':
            selected_base_classifiers = ['ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC']
            self.run_pu_learning(selected_base_classifiers=selected_base_classifiers,
                                 final_level_classifier=self.final_level_classifier)
        else:
            self.run_pu_learning()


if __name__ == '__main__':

    config_file = '../../../config.yaml'
    cfg = Config(config_file)

    data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')
    print('nthreads:', cfg.nthreads)
    print('Stochastic iterations:', cfg.iterations)

    clf_id = 'ExtraTreesClassifier'
    pu = PULearning(cfg, data, clf_id)
    pu.run()
