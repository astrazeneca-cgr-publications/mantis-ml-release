import pandas as pd
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb

from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.prepare_train_test_sets import PrepareTrainTestSets
from mantis_ml.modules.supervised_learn.classifiers.generic_classifier import GenericClassifier
from mantis_ml.modules.supervised_learn.classifiers.ensemble_lib import ensemble_clf_params
from mantis_ml.modules.supervised_learn.core.ml_plot_functions import plot_feature_imp_for_classifier


class SklearnWrapper(object):
    '''
    Extend the Sklearn module to invoke in-built methods seamlessly
    across multiple sklearn classifiers.
    '''
    def __init__(self, clf, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def evaluate(self, x, y, verbose=0):
        self.clf.evaluate(x, y, verbose)

    def feature_importances(self, x, y):
        return (self.clf.fit(x, y).feature_importances_)

    def get_coef_(self):
        return self.clf.coef_



class SklearnExtendedClassifier(GenericClassifier):

    def __init__(self, cfg, X_train, y_train, X_test, y_test,
                 train_gene_names, test_gene_names, clf_id, clf_params, make_plots, verbose=False):

        GenericClassifier.__init__(self, cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, clf_id, verbose)

        self.clf_id = clf_id
        self.clf_params = clf_params
        self.make_plots = make_plots
        self.verbose = verbose

    def build_model(self):
        self.model = SklearnWrapper(clf=eval(self.clf_id), params=self.clf_params)

    def evaluate_model(self):
        # dummy placeholders: -1
        self.train_acc = -1
        self.test_acc = -1
        
        y_test_roc = np.argmax(self.y_test, axis=1)
        y_pred_roc = self.y_prob[:, 1]
        self.auc_score = roc_auc_score(y_test_roc, y_pred_roc)
        print(f"AUC: {self.auc_score}")

    def get_feature_importance(self, X_train, y_train, feature_cols, clf_id):
        cur_feature_imp = self.model.feature_importances(X_train, y_train)

        self.feature_dataframe = pd.DataFrame({'features': feature_cols, clf_id: cur_feature_imp})
        #print(self.feature_dataframe)

    def process_predictions(self):
        self.evaluate_model()
        if self.make_plots:
            self.plot_confusion_matrix(verbose=False)
            self.plot_roc_curve(make_plot=True)
        self.aggregate_predictions()



if __name__ == '__main__':

    config_file = '../../../config.yaml'
    cfg = Config(config_file)

    set_generator = PrepareTrainTestSets(cfg)

    data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')
    train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(data)

    # select random balanced dataset
    i = random.randint(0, len(train_dfs)-1)
    print(f"i: {i}")
    train_data = train_dfs[i]
    test_data = test_dfs[i]
    print(f"Training set size: {train_data.shape[0]}")
    print(f"Test set size: {test_data.shape[0]}")

    X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data,
                                                                                                    test_data)

    make_plots = True
    verbose = False

    clf_id = 'ExtraTreesClassifier'     # 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'XGBoost'
    clf_params = ensemble_clf_params[clf_id]

    model = SklearnExtendedClassifier(cfg, X_train, y_train, X_test, y_test,
                                      train_gene_names, test_gene_names, clf_id, clf_params, make_plots, verbose)

    if clf_id == 'XGBoost':
        model.model = xgb.XGBClassifier(**clf_params)
    else:
        model.build_model()


    model.train_model()

    # get feature importance
    feature_cols = train_data.drop([cfg.Y], axis=1).columns.values
    model.get_feature_importance(X_train, y_train, feature_cols, clf_id)
    plot_feature_imp_for_classifier(model.feature_dataframe, clf_id, clf_id, cfg.superv_feat_imp)

    model.process_predictions()

