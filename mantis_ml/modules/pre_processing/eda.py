import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mantis_ml.config_class import Config


class EDA:

    def __init__(self, cfg):
        self.cfg = cfg


    def read_input_feature_table(self):
        '''
        Read input data
        
        :return: 
        '''
        print('\n>> Reading input table...')
        df = pd.read_csv(self.cfg.complete_feature_table, sep='\t')

        gene_names = df['Gene_Name']
        df.drop(['Gene_Name'], axis=1, inplace=True)

        print("Total number of features: {0}\n".format(str(df.shape[1])))

        return df, gene_names


    def drop_duplicate_features(self, df):
        '''
        Drop duplicate features
        
        :param df: 
        :return: 
        '''
        print('\n>> Removing duplicate features...')

        corr_matrix = df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation == 1.0
        drop_elements = [column for column in upper.columns if any(upper[column] == 1)]
        print('Duplicate features:', len(drop_elements))
        print(drop_elements)

        df = df.drop(drop_elements, axis=1)

        print("-Remaining number of features: {0}".format(str(df.shape[1])))
        print('---------')

        return df


    def drop_uninformative_features(self, df):
        '''
        Drop non-informative / biased features (included in the default feature space)
        
        :param df: 
        :return: 
        '''
        print('\n>> Dropping uninformative features...')

        drop_elements = [] 
        if self.cfg.drop_gene_len_features:
            drop_elements = ['GeneSize', 'ExAC_cds_len', 'ExAC_gene_length']

        zero_sum_cols = df.columns[(df == 0).all()].values
        drop_elements.extend(zero_sum_cols)

        print('Discarded {0} uninformative feature(s)'.format(str(len(drop_elements))))
        print(drop_elements)
        df = df.drop(drop_elements, axis=1, errors='ignore')

        print("-Remaining number of features: {0}".format(str(df.shape[1])))

        return df


    def plot_feature_corr_map(self, df, filename_suffix, annot=True):
        '''
        Plot Pearson's correlation heatmap
        
        :param df: 
        :param filename_suffix: 
        :return: 
        '''
        if self.cfg.phenotype == 'Generic':
            return 0

        print('\n>> Plotting feature correlation map (' + filename_suffix + ')...')
        colormap = plt.cm.RdBu
        if self.cfg.generic_classifier:
            annot = False

        plt.figure(figsize=(32, 28))
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        sns_plot = sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1.0,
                               square=True, cmap=colormap, linecolor='white', annot=annot, annot_kws={"size": 7})

        plot_filepath = str(self.cfg.eda_out / ("feature_correlations." + filename_suffix + ".pdf"))
        sns_plot.get_figure().savefig(plot_filepath, format='pdf', bbox_inches='tight')


    def fix_feature_data_types(self, df):
        '''
        Fix type of features included by default
        
        :param df: 
        :return: 
        '''
        print('\n>> Fixing feature types...')

        # Inspect numeric features
        num_df = df.select_dtypes(exclude=['object']).copy()
        # num_df.head(10)


        numeric_to_str_elements = [self.cfg.Y]

        for col in numeric_to_str_elements:
            df[col] = df[col].apply(str)

        # Inspect non-numeric (object) features
        obj_df = df.select_dtypes(include=['object']).copy()
        # obj_df.head()

        if 'GWAS_trait' in df.columns.values:
            df['GWAS_trait'].fillna(0, inplace=True)
            df.loc[df.GWAS_trait != 0, 'GWAS_trait'] = 1

        # df['collapsed_GO'] = df.Renal_GO_Ontology.apply(int) + df.Kidney_GO_Ontology.apply(int) + df.Nephro_GO_Ontology.apply(int) + df.Glomerular_GO_Ontology.apply(int) + df['Distal Tubule_GO_Ontology'].apply(int)


        # Convert probability scores to phred scores
        cols_with_pvals = []
        proba_cols = ['ExAC_mu_lof', 'ExAC_pLI', 'ExAC_pNull', 'LoF_FDR_ExAC']

        all_pval_cols = cols_with_pvals
        for col in all_pval_cols:
            if col not in df.columns:
                continue

            df[col] = -10 * np.log10(df[col])

            max_thres = -10 * np.log10(10 ** (-8))
            df.loc[df[col] > max_thres, col] = max_thres

            if df[col].isnull().sum() != 0:
                # TODO: replace with 0 or 1?
                df[col].fillna(0, inplace=True)
                print("[Warning]: NA values in feature '" + col + "' were replaced by zeros.")

        return df


    def check_for_missing_data(self, df):
        '''
        Verify that there are no features with missing values
        :param df: 
        :return: 
        '''
        print(">> Checking for missing data...")
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        if missing_data['Total'].sum() == 0:
            print('All missing data have been successfully imputed.')
        else:
            print('[Warning]: There are still features with missing data:')
            print(missing_data)


    def drop_elems_w_high_corr(self, df, thres=0.85):
        '''
        Drop elements with corr ≥ threshold (default: 0.85)
        
        :param df: 
        :param thres: 
        :return: 
        '''
        print('\n>> Removing features with high correlation...')

        # Create correlation matrix
        corr_matrix = df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than thres
        to_drop = [column for column in upper.columns if any(upper[column] > thres)]

        clean_df = df.drop(to_drop, axis=1)
        print("Removed {0} features with correlation > {1} ...".format(str(len(to_drop)), str(thres)))
        print('-Remaining number of features: {0}'.format(str(clean_df.shape[1])))

        return clean_df


    def drop_linearly_dependent_features(self, df):
        # TODO: incomplete function
        print('\n>> Removing linearly dependent features using QR decomposition...')
        # reduced_form, inds = sympy.Matrix(df.values).rref()

        numeric_cols = df.select_dtypes(include=['int32', 'int64', 'float64']).columns.values
        non_numeric_cols = df.select_dtypes(exclude=['int32', 'int64', 'float64']).columns.values

        num_df = df[numeric_cols]
        non_num_df = df[non_numeric_cols]

        qr_df = np.linalg.qr(num_df)[1]
        print(qr_df)
        print(qr_df.sum(axis=1)) # those with sum==0 should be filtered out
        # print(df.iloc[:, inds].head())


    def plot_hist_for_numeric_features(self, df, pos_label="known_genes", neg_label="unrelated"):
        '''
        Plot histograms for all numeric features
    
        :param df: 
        :param pos_label: 
        :param neg_label: 
        :return: 
        '''
        if self.cfg.phenotype == 'Generic':
            return 0

        print('\n>> Plotting histograms for all numeric features...')

        # Plot histograms for all numeric features
        numeric_cols = df.select_dtypes(include=['int32', 'int64', 'float64']).columns.values
        # make sure Y-label column is not part of the features to examine
        numeric_cols = [c for c in numeric_cols if c != self.cfg.Y]

        if len(numeric_cols) == 0:
            return 0

        fig_cols = 4
        fig_rows = int(len(numeric_cols) / fig_cols) + 1

        fig, ax = plt.subplots(figsize=(23, 34))

        cols_for_log_plus_one = ['GTEx_Kidney_TPM_expression']

        for i, j in itertools.zip_longest(numeric_cols, range(len(numeric_cols))):
            plt.subplot(fig_rows, fig_cols, j + 1)
            plt.subplots_adjust(wspace=0.3, hspace=0.5)

            tmp_df = df[[i, self.cfg.Y]]
            if i in cols_for_log_plus_one:
                tmp_df.iloc[:, 0] = np.log(df[i] + 1)


            pos_tmp_df = tmp_df.loc[tmp_df[self.cfg.Y].astype(float) == 1, i]
            pos_tmp_df.rename(pos_label, inplace=True)

            neg_tmp_df = tmp_df.loc[tmp_df[self.cfg.Y].astype(float) == 0, i]
            neg_tmp_df.rename(neg_label, inplace=True)

            sns.kdeplot(pos_tmp_df, shade=True, color="#fb6a4a")
            sns.kdeplot(neg_tmp_df, shade=True, color="#3182bd")
            plt.title(i)

        plt.show()

        plot_filepath = str(self.cfg.eda_out / 'numerical_features_histograms.pdf')
        fig.savefig(plot_filepath, format='pdf', bbox_inches='tight')


    def plot_hist_for_non_numerical_features(self, df, pos_label="known_genes", neg_label="unrelated"):
        '''
        Plot histograms for all categorical features
        
        :param df: 
        :param pos_label: 
        :param neg_label: 
        :return: 
        '''
        if self.cfg.phenotype == 'Generic':
            return 0

        print('\n>> Plotting histograms for all categorical features...')

        # Plot histograms for all non-numeric features
        non_numeric_cols = df.select_dtypes(exclude=['int32', 'int64', 'float64']).columns.values
        # make sure Y-label column is not part of the features to examine
        non_numeric_cols = [c for c in non_numeric_cols if c != self.cfg.Y]

        if len(non_numeric_cols) == 0:
            return 0
        #     print(non_numeric_cols)

        fig, ax = plt.subplots(1, 2, figsize=(18, 10))

        for i, j in itertools.zip_longest(non_numeric_cols, range(len(non_numeric_cols))):
            #         _ = plt.subplot(7, 3, j+1)
            #         _ = plt.subplots_adjust(wspace=0.3, hspace=0.5)

            tmp_df = df[[i, self.cfg.Y]]

            pos_tmp_df = tmp_df.loc[tmp_df[self.cfg.Y].astype(float) == 1, i]
            _ = pos_tmp_df.rename(pos_label, inplace=True)

            neg_tmp_df = tmp_df.loc[tmp_df[self.cfg.Y].astype(float) == 0, i]
            _ = neg_tmp_df.rename(neg_label, inplace=True)

            _ = plt.title(i)
            _ = sns.countplot(pos_tmp_df, color="#fb6a4a", ax=ax[0])
            _ = sns.countplot(neg_tmp_df, color="#3182bd", ax=ax[1])

        plt.show()

        plot_filepath = str(self.cfg.eda_out / 'categorical_features_histograms.pdf')
        fig.savefig(plot_filepath, format='pdf', bbox_inches='tight')


    def standardise_features(self, df):
        '''
        Feature standardisation
        
        :param df: 
        :return: 
        '''
        print('\n>> Normalising feature table...')

        object_cols = df.select_dtypes(include=['object']).columns.values

        elements_to_standardise = df.select_dtypes(include=['int32', 'int64', 'float64']).columns.values
        features_standard = StandardScaler().fit_transform(df[elements_to_standardise])

        std_df = pd.DataFrame(features_standard, columns=elements_to_standardise)

        std_df = pd.concat([std_df, df[object_cols]], axis=1)

        return std_df


    def save_standardised_feat_table_to_file(self, std_df):
        '''
        Save standardised feature table to file
        
        :param std_df: 
        :return: 
        '''
        filepath = str(self.cfg.processed_data_dir / "standardised_feature_table.tsv")
        std_df.to_csv(filepath, sep='\t', index=False)


    def dummify_categorical_vars(self, std_df):
        '''
        One-Hot encoding of categorical variables
    
        :param std_df: 
        :return: dummy_df 
        '''
        print('\n>> Dummifying categorical variables (one-hot encoding)...')

        dummy_df = pd.get_dummies(std_df.loc[:, std_df.columns != self.cfg.Y])
        dummy_df[self.cfg.Y] = std_df[self.cfg.Y].copy()

        return dummy_df


    def select_features_manually(self, df):
        '''
        Bespoke feature selection for PCA optimisation
        :param df: 
        :return: 
        '''
        cols_to_keep = ['RVIS', 'RVIS_ExAC', 'MTR_ExACv2', 'ExAC_pREC', 'ExAC_pLI', 'LoF_FDR_ExAC',
                        'GTEx_Kidney_TPM_expression', 'MGI_mouse_knockout_feature', 'glomerular_expr_flag',
                        'Experimental_perc_core_overlap', 'Inferred_perc_core_overlap',
                        'DAPPLE_perc_core_overlap', 'CKDdb_num_of_studies', 'GOA_Kidney_Research_Priority',
                        'GO_kidney', 'GO_renal', 'GO_glomerul', 'known_gene',
                        'ExAC_gene_length', 'ExAC_dup.score', 'ExAC_del.score',
                        # 'ExAC_mu_lof', 'ExAC_dip',
                        'ExAC_n_lof', 'ExAC_lof_z',
                        'ExAC_pNull', 'geneCov_ExACv2',
                        'ExAC_mean_rd', 'ExAC_gc_content', 'ExAC_complexity', 'ExAC_segdups',
                        'ExAC_flag', 'essential_mouse_knockout', 'non_essential_mouse_knockout',
                        'GWAS_tissue_trait_flag', 'GTEx_Kidney_Expression_Rank',
                        'ProteinAtlas_RNA_expression_TMP', 'Kidney_GWAS_hits', 'Kidney_GWAS_tissue_trait_flag',
                        # 'ProteinAtlas_gene_expr_level_High', 'ProteinAtlas_gene_expr_level_Low',
                        # 'ProteinAtlas_gene_expr_level_Medium', 'ProteinAtlas_gene_expr_level_Not detected'
                        ]

        # Boruta - CKD confirmed features
        cols_to_keep = ['GO_renal', 'ExAC_mis_z', 'GO_kidney', 'LoF_FDR_ExAC', 'glomerular_expr_flag',
                        'GTEx_Kidney_Expression_Rank', 'GWAS_hits', 'CKDdb_num_of_studies', 'mut_prob_all',
                        'ExAC_n_lof', 'Inferred_perc_core_overlap', 'ExAC_pREC', 'ExAC_pNull', 'ExAC_pLI',
                        'ExAC_lof_z', 'GTEx_Kidney_TPM_expression', 'ExAC_cds_len', 'GeneSize',
                        'ProteinAtlas_RNA_expression_TMP', 'essential_mouse_knockout', 'GOA_Kidney_Research_Priority',
                        'MGI_essential_gene', 'MGI_mouse_knockout_feature', 'known_gene']


        # - Most likely exclude: NEPHRO, RVIS_ExACv2 (highly correlated)
        # - May need to exclude: GWAS_hits
        # - Probably discard: 'tub_Exp_num_of_eQTLs', 'tub_Pr_of_no_eQTL', 'glom_Exp_num_of_eQTLs', 'glom_Pr_of_no_eQTL', 'CKDdb_Disease'
        # - Discard: ExAC_mu_syn, ExAC_mu_mis, ExAC_n_syn, ExAC_n_mis, ExAC_exp_lof,
        #            ExAC_exp_mis, ExAC_exp_syn, ExAC_mis_z, ExAC_syn_z, ExAC_mis_z,
        #            ExAC_num_targ, ExAC_del, ExAC_dup, ExAC_del.sing, ExAC_dup.sing
        #            ExAC_del.sing.score, ExAC_dup.sing.score, ExAC_cnv.score,
        #            mut_prob_syn, mut_prob_mis, mut_prob_non, mut_prob_splice_site,
        #            mut_prob_frameshift, tub_FDR, glom_FDR, glomerular_expr_flag, tubular_expr_flag

        df = df[cols_to_keep]

        return df


    def save_feature_table_to_file(self, df, gene_names, file_descr='processed'):
        '''
        Write processed feature table to file
    
        :param df: 
        :param gene_names: 
        :param file_descr: 
        :return: 
        '''
        df['Gene_Name'] = gene_names

        out_file = file_descr + '_feature_table.tsv'
        df.to_csv(self.cfg.processed_data_dir / out_file, sep='\t', index=False)


    def fix_classes_imbalance(self, df, gene_names, balancing_ratio=1.5, random_state=0):
        '''
        Fix imbalance of Positive and Negative classes
    
        :param df: 
        :param gene_names: 
        :param balancing_ratio: 
        :param random_state: 
        :return: 
        '''
        print('\n>> Fixing class imbalance...')

        df['Gene_Name'] = gene_names
        known_df = df.loc[df[self.cfg.Y].astype(float) == 1]
        unrelated_df = df.loc[df[self.cfg.Y].astype(float) == 0]
        unrelated_df = unrelated_df.sample(n=round(known_df.shape[0] * balancing_ratio), random_state=random_state)

        balanced_df = pd.concat([known_df, unrelated_df], axis=0)

        print('Balanced feature table: {0} data points x {1} features.'.format(str(balanced_df.shape[0]), str(
            balanced_df.shape[1])))

        return balanced_df


    def split_into_train_and_test_sets(self, df, test_size=0.2, random_state=0, prefix='balanced'):
        '''
        Split data into training and test set
    
        :param df: 
        :param test_size: 
        :param random_state: 
        :param prefix: 
        :return: 
        '''
        print('\n>> Splitting _{0}_ dataset to train and test sets...'.format(prefix))

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[self.cfg.Y])

        print("Training set size: " + str(len(train_df)))
        print("Test set size: " + str(len(test_df)))

        train_file = prefix + '_train_df.tsv'
        test_file = prefix + '_test_df.tsv'

        train_df.to_csv(self.cfg.processed_data_dir / train_file, sep='\t', index=False)
        test_df.to_csv(self.cfg.processed_data_dir / test_file, sep='\t', index=False)

        return train_df, test_df
