import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import xgboost as xgb
from keras.optimizers import Adam
from keras.utils import to_categorical


from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.prepare_train_test_sets import PrepareTrainTestSets
from mantis_ml.modules.supervised_learn.classifiers.ensemble_lib import (get_oof,
                                                                         ensemble_clf_params)
from mantis_ml.modules.supervised_learn.core.ml_plot_functions import plot_average_feature_importance_scatterplots
from mantis_ml.modules.supervised_learn.classifiers.sklearn_extended_classifier import SklearnWrapper, SklearnExtendedClassifier
from mantis_ml.modules.supervised_learn.classifiers.dnn import DnnClassifier



class EnsembleClassifier():

    def __init__(self, cfg, base_sklearn_clf_params, final_clf_params):

        self.cfg = cfg
        self.base_sklearn_clf_params = base_sklearn_clf_params
        self.final_clf_params = final_clf_params

        self.base_clf_ids = sorted(base_sklearn_clf_params.keys())
        self.final_clf_ids = sorted(final_clf_params.keys())

    def build_base_classifiers(self):
        '''
        Build base classifier objects for the Ensemble classifier
        '''
        print("\n>> Building base classifiers for Ensemble classifier...")
        self.base_classifiers = {}

        # create sklearn classifier objects
        for clf_id in self.base_clf_ids:
            #print(clf_id)
            tmp_params = self.base_sklearn_clf_params[clf_id]
            tmp_clf = SklearnWrapper(clf=eval(clf_id), params=tmp_params)
            self.base_classifiers[clf_id] = tmp_clf
        #print(self.base_classifiers)

    def get_feature_imp_per_base_clf(self, X_train, y_train, feature_cols):
        '''
        Get feature importance score per base classifier
        :param X_train: 
        :param y_train: 
        :param feature_cols: 
        :return: 
        '''
        print("\n>>Getting feature importance score per base classifier...")
        self.feature_imp_per_clf = {}

        for clf_id in self.base_clf_ids:
            if clf_id in ['SVC']:
                continue
            print(clf_id)
            cur_clf = self.base_classifiers[clf_id]

            cur_feature_imp = cur_clf.feature_importances(X_train, y_train)
            self.feature_imp_per_clf[clf_id] = cur_feature_imp
            print("Extracted feature importance for [" + clf_id + "]")

        self.feature_dataframe = pd.DataFrame.from_dict(self.feature_imp_per_clf)
        self.feature_dataframe['features'] = feature_cols

    def plot_feature_imp_scores(self, feature_dataframe):
        # for clf_id in self.base_clf_ids:
        #     if clf_id in ['SVC']:
        #         continue
        #     plot_feature_imp_for_classifier(self.feature_dataframe, clf_id, clf_id + ' Feature Importance')
        plot_average_feature_importance_scatterplots(feature_dataframe, self.cfg.superv_feat_imp)


    def get_base_level_predictions(self, X_train, y_train, X_test):
        '''
        Get base-level predictions to be used as additional features for second-level classifier
        '''
        print("\n>>Getting base level out-of-fold predictions...")
        self.oof_base_predictions = {}

        for clf_id in self.base_clf_ids:
            #print(clf_id)
            cur_clf = self.base_classifiers[clf_id]

            clf_oof_train, clf_oof_test = get_oof(cur_clf, X_train, y_train, X_test)
            print("[" + clf_id + "] Training is complete")

            self.oof_base_predictions[clf_id] = [clf_oof_train, clf_oof_test]

        print("\n>>First-Level Training is complete")


    # def get_performace_by_classifier(self, X_train, y_train, X_test, y_test, test_gene_names):
    #
    #     for clf_id in stack_clf.base_clf_ids:
    #         print(clf_id)
    #         cur_clf = self.base_classifiers[clf_id]
    #         get_clf_performance(cur_clf, clf_id, X_train, y_train, X_test, y_test, test_gene_names)


    def concat_first_lvl_predictions(self):
        final_lvl_X_train = []
        final_lvl_X_test = []

        for clf, arr in self.oof_base_predictions.items():
            final_lvl_X_train.append(self.oof_base_predictions[clf][0])
            final_lvl_X_test.append(self.oof_base_predictions[clf][1])

        final_lvl_X_train = np.array(final_lvl_X_train)
        final_lvl_X_train = final_lvl_X_train.reshape(-1, final_lvl_X_train.shape[1]).T
        #print(final_lvl_X_train.shape)

        final_lvl_X_test = np.array(final_lvl_X_test)
        final_lvl_X_test = final_lvl_X_test.reshape(-1, final_lvl_X_test.shape[1]).T
        #print(final_lvl_X_test.shape)

        return final_lvl_X_train, final_lvl_X_test


    def run_final_level_classifier(self, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names):

        clf_id = self.final_clf_params.keys()
        clf_id = list(clf_id)[0]

        print('> Final layer training')
        print('Classifier:', clf_id)

        self.final_clf_params = self.final_clf_params[clf_id]
        self.final_clf_params['clf_id'] = clf_id
        # ======================

        self.final_model = None
        if clf_id == 'DNN':
            # convert target variables to 2D arrays
            y_train = np.array(to_categorical(y_train, 2))
            y_test = np.array(to_categorical(y_test, 2))

            self.final_model = DnnClassifier(self.cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, **self.final_clf_params)
            self.final_model.run()
        elif clf_id == 'XGBoost':
            make_plots = False
            self.final_model = SklearnExtendedClassifier(self.cfg, X_train, y_train, X_test, y_test,
                                              train_gene_names, test_gene_names,
                                              clf_id, self.final_clf_params, make_plots)
            self.final_model.model = xgb.XGBClassifier(**self.final_clf_params)
            print(X_train.shape)
            print(y_train.shape)

            self.final_model.train_model()
            self.final_model.process_predictions()


        # assign results of final-level classifier to the whole ensemble classifier object
        self.y_prob = self.final_model.y_prob

        self.train_acc = self.final_model.train_acc
        self.test_acc = self.final_model.test_acc
        self.auc_score = self.final_model.auc_score

        self.tp_genes = self.final_model.tp_genes
        self.fp_genes = self.final_model.fp_genes

        self.tn_genes = self.final_model.tn_genes
        self.fn_genes = self.final_model.fn_genes



if __name__ == '__main__':

    config_file = '../../../config.yaml'
    cfg = Config(config_file)

    set_generator = PrepareTrainTestSets(cfg)
    # read processed feature table
    data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')

    train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(data, stratified_kfold=False)


    # use random balanced dataset
    i = 10 #random.randint(0, len(train_dfs))

    print(f"\n> Iteration {i}")
    train_data = train_dfs[i]
    test_data = test_dfs[i]
    print(f"Training set size: {train_data.shape[0]}")
    print(f"Test set size: {test_data.shape[0]}")

    print(train_data.head())
    print(test_data.head())


    X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data,
                                                                                                    test_data)

    # Define Stacking classifiers
    selected_base_classifiers = ['ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC']
    final_level_classifier = 'DNN' #'XGBoost'


    base_sklearn_clf_params = dict((k, v) for k, v in ensemble_clf_params.items() if k in selected_base_classifiers)
    final_clf_params = {final_level_classifier: ensemble_clf_params[final_level_classifier]}




    # **************** Create Stacking classifier instance **********************
    stack_clf = EnsembleClassifier(cfg, base_sklearn_clf_params, final_clf_params)

    stack_clf.build_base_classifiers()

    # """
    # Extract feature importance
    feature_cols = train_data.drop([cfg.Y], axis=1).columns.values
    stack_clf.get_feature_imp_per_base_clf(X_train, y_train, feature_cols)
    print(stack_clf.feature_dataframe)

    # stack_clf.plot_feature_imp_scores(stack_clf.feature_dataframe)

    # [Not Necessary for Production] Check performance of individual classifiers
    # stack_clf.get_performace_by_classifier(X_train, y_train, X_test, y_test, test_gene_names)
    # """

    # stacking
    stack_clf.get_base_level_predictions(X_train, y_train, X_test)

    # concatenate first level predictions to feed to the final layer
    final_lvl_X_train, final_lvl_X_test = stack_clf.concat_first_lvl_predictions()


    # TODO: use this section for debugging only
    if False:
        # add confirmed features from Boruta
        ckd_boruta_confirmed_features = ['GO_renal', 'ExAC_mis_z', 'GO_kidney', 'LoF_FDR_ExAC', 'glomerular_expr_flag',
                                         'GTEx_Kidney_Expression_Rank', 'GWAS_hits', 'CKDdb_num_of_studies', 'mut_prob_all',
                                         'ExAC_n_lof', 'Inferred_perc_core_overlap', 'ExAC_pREC', 'ExAC_pNull', 'ExAC_pLI',
                                         'ExAC_lof_z', 'GTEx_Kidney_TPM_expression', 'ExAC_cds_len', 'GeneSize',
                                         'ProteinAtlas_RNA_expression_TMP', 'essential_mouse_knockout',
                                         'GOA_Kidney_Research_Priority', 'MGI_essential_gene', 'MGI_mouse_knockout_feature']

        confirmed_feature_indexes = sorted([list(train_data.columns.values).index(f) for f in ckd_boruta_confirmed_features])

        final_lvl_X_train = np.concatenate((X_train[:, confirmed_feature_indexes], final_lvl_X_train), axis=1)
        final_lvl_X_test = np.concatenate((X_test[:, confirmed_feature_indexes], final_lvl_X_test), axis=1)

    stack_clf.run_final_level_classifier(final_lvl_X_train, y_train, final_lvl_X_test, y_test, train_gene_names, test_gene_names)
