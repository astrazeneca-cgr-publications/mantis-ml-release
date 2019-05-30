from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import random
import sys
from mantis_ml.config_class import Config


class BalancedDatasetRandomGenerator:

    def __init__(self, cfg, df, Y, balancing_ratio, kfold, random_state, stratified_kfold=True):
        self.cfg = cfg
        self.df = df
        self.Y = Y
        self.balancing_ratio = balancing_ratio
        self.kfold = kfold
        self.random_state = random_state
        self.stratified_kfold = stratified_kfold

    def get_balanced_datasets(self, verbose=False):
        '''
        Create random partition of balanced datasets covering the entire original dataset
        :param verbose: 
        :return: 
        '''

        print("\n>> Getting balanced datasets...")
        all_positive_df = self.df.loc[self.df[self.Y] == 1]

        unrelated_df = self.df.loc[self.df[self.Y] == 0]
        total_neg_size = unrelated_df.shape[0]
        neg_sample_size = unrelated_df.shape[0]

        self.balanced_dfs_list = []

        while unrelated_df.shape[0] >= neg_sample_size:
            # TODO: Beta -- get random susample from positive genes
            positive_df = all_positive_df.sample(n=round(len(all_positive_df) * self.cfg.seed_pos_ratio),
                                                 random_state=random.randint(0, 1000000))

            neg_sample_size = round(positive_df.shape[0] * self.balancing_ratio)

            balanced_unrelated_df = unrelated_df.sample(n=neg_sample_size, random_state=self.random_state)
            unrelated_df = unrelated_df.drop(balanced_unrelated_df.index, axis=0)

            balanced_df = pd.concat([positive_df, balanced_unrelated_df], axis=0)
            self.balanced_dfs_list.append(balanced_df)

        # Allocate remaining points to each of the subsamples in the balanced_dfs_list
        rem_sample_size = round(unrelated_df.shape[0] / len(self.balanced_dfs_list)) + 1

        list_idx = -1
        while unrelated_df.shape[0] > rem_sample_size:
            list_idx += 1
            rem_unrelated_df = unrelated_df.sample(n=rem_sample_size, random_state=self.random_state)

            unrelated_df = unrelated_df.drop(rem_unrelated_df.index, axis=0)
            self.balanced_dfs_list[list_idx] = pd.concat([self.balanced_dfs_list[list_idx], rem_unrelated_df], axis=0)

        self.balanced_dfs_list[list_idx] = pd.concat([self.balanced_dfs_list[list_idx], unrelated_df], axis=0)


        if self.cfg.seed_pos_ratio == 1:
            print('>>>>>> Data Sanity Check <<<<<<<')
            total_rows = sum([m.shape[0] for m in self.balanced_dfs_list])
            verif_rows = all_positive_df.shape[0] * len(self.balanced_dfs_list) + total_neg_size

            assert verif_rows == total_rows, 'Validation Error: [verif_rows != total_rows] - ' \
                                         'not all negative data points have been included in a balanced subsample.'
            print('Passed.')

        if verbose:
            for l in self.balanced_dfs_list:
                print('Balanced sample feature table: ' + str(l.shape[0]) + ' data points x ' + str(l.shape[1]) + ' features.')


    def split_into_train_and_test_sets(self, verbose=False):
        '''
        Split each (balanced) dataset into training and test sets
        :param verbose: 
        :return: 
        '''
        print("\n>> Splitting into training and test sets...")

        self.train_dfs = []
        self.test_dfs = []

        # Starified K-fold
        if self.stratified_kfold:
            for bal_df in self.balanced_dfs_list:

                k_fold_random_state = random.randint(0, 1000000)
                skf = StratifiedKFold(n_splits=self.kfold, random_state=k_fold_random_state, shuffle=False) #self.random_state, shuffle=False)

                for train_index, test_index in skf.split(bal_df, bal_df[self.cfg.Y]):
                    if verbose:
                        print('Train size:', len(train_index), ', Test size:', len(test_index))
                    train_df, test_df = bal_df.iloc[train_index], bal_df.iloc[test_index]

                    self.train_dfs.append(train_df)
                    self.test_dfs.append(test_df)
        else:
            for bal_df in self.balanced_dfs_list:
                for iter in range(self.cfg.kfold):

                    fold_random_state = self.random_state
                    if self.cfg.random_fold_split:
                        fold_random_state = random.randint(0, 1000000000)
                    train_df, test_df = train_test_split(bal_df, test_size=self.cfg.test_size,
                                                                  random_state=fold_random_state,
                                                                  stratify=bal_df[self.Y])
                    self.train_dfs.append(train_df)
                    self.test_dfs.append(test_df)

        print(f"\nTotal num of train/test sets: {len(self.train_dfs)}")

        return self.train_dfs, self.test_dfs


if __name__ == '__main__':

    config_file = '../../../config.yaml'
    cfg = Config(config_file)

    data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')
    print(data.head())

    verbose = True

    print(f"Random state: {cfg.random_state}")
    bal_sample_gen = BalancedDatasetRandomGenerator(cfg, data, cfg.Y, cfg.balancing_ratio, cfg.kfold, cfg.random_state)
    bal_sample_gen.get_balanced_datasets(verbose=verbose)
    bal_sample_gen.split_into_train_and_test_sets(verbose=verbose)