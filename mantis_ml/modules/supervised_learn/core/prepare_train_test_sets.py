import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from mantis_ml.modules.supervised_learn.core.balanced_datasets_random_generator import BalancedDatasetRandomGenerator


class PrepareTrainTestSets:

    def __init__(self, cfg):
        self.cfg = cfg

    def get_balanced_train_test_sets(self, data, stratified_kfold=True, verbose=False, random_state=None):

        if random_state is None:
            random_state = self.cfg.random_state

        bal_sample_gen = BalancedDatasetRandomGenerator(self.cfg, data, self.cfg.Y, self.cfg.balancing_ratio, self.cfg.kfold,
                                                        random_state, stratified_kfold=stratified_kfold)

        bal_sample_gen.get_balanced_datasets(verbose=verbose)
        train_dfs, test_dfs = bal_sample_gen.split_into_train_and_test_sets(verbose=verbose)

        return train_dfs, test_dfs


    def prepare_train_test_tables(self, train_data, test_data):
        train_gene_names = train_data[self.cfg.gene_name]
        train_data.drop(self.cfg.gene_name, axis=1, inplace=True)

        test_gene_names = test_data[self.cfg.gene_name]
        test_data.drop(self.cfg.gene_name, axis=1, inplace=True)

        X_train = np.array(train_data.drop(self.cfg.Y, axis=1))
        y_train = train_data[self.cfg.Y].ravel()

        X_test = np.array(test_data.drop(self.cfg.Y, axis=1))
        y_test = test_data[self.cfg.Y].ravel()

        return X_train, y_train, train_gene_names, X_test, y_test, test_gene_names