import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import sys

from mantis_ml.config_class import Config

feature_imp_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'XGBoost']
gene_colors = {'Known': '#bdbdbd', 'Novel': '#31a354', 'Highlighted': 'red'}


class IterationsBenchmark():

    def __init__(self, cfg, clif_id, iterations):
        self.cfg = cfg
        self.clf_id = clif_id
        self.iterations = iterations

        try:
            proc_feat_df = pd.read_csv(self.cfg.processed_data_dir / 'processed_feature_table.tsv', sep='\t')
            self.ordered_features = [f for f in proc_feat_df.columns.values if f not in ['known_gene', 'Gene_Name']]

            tmp_eval_metrics_df = pd.read_csv(self.cfg.superv_out / ('PU_' + self.clf_id + '.evaluation_metrics.tsv'), sep='\t')
            self.total_runs = tmp_eval_metrics_df.shape[0]
            del tmp_eval_metrics_df
        except:
            print('Non defined classifier: Instantiated only basic ClassifierEvaluator skeleton.')

        self.known_genes_df = proc_feat_df[['Gene_Name', 'known_gene']]
        self.known_genes = self.known_genes_df.loc[self.known_genes_df.known_gene == 1, 'Gene_Name'].values




    def get_definitive_gene_predictions(self, pos_ratio_thres=0.99):

        self.gene_pred_df = pd.read_csv(self.cfg.out_root / ('../../benchmarking-for-iterations/PU_' + self.clf_id + '.all_genes_predictions.tsv.iterations_' + str(self.iterations)), sep='\t')
        self.gene_pred_df.columns.values[0] = 'Gene_Name'
        self.gene_pred_df = pd.merge(self.gene_pred_df, self.known_genes_df, left_on='Gene_Name', right_on='Gene_Name',
                                     how='outer')

        self.gene_pred_df['pos_ratio'] = self.gene_pred_df['positive_genes'] / (
        self.gene_pred_df['positive_genes'] + self.gene_pred_df['negative_genes'])
        self.gene_pred_df.sort_values(by=['pos_ratio'], ascending=False, inplace=True)
        # print(self.gene_pred_df.head())

        self.predicted_genes_df = self.gene_pred_df.loc[self.gene_pred_df.pos_ratio >= pos_ratio_thres]
        self.predicted_known_genes_df = self.gene_pred_df.loc[
            (self.gene_pred_df.pos_ratio >= pos_ratio_thres) & (self.gene_pred_df.known_gene == 1)]
        self.predicted_novel_genes_df = self.gene_pred_df.loc[
            (self.gene_pred_df.pos_ratio >= pos_ratio_thres) & (self.gene_pred_df.known_gene == 0)]

        self.predicted_genes = sorted(self.predicted_genes_df['Gene_Name'].values)
        self.predicted_known_genes = sorted(self.predicted_known_genes_df['Gene_Name'].values)
        self.predicted_novel_genes = sorted(self.predicted_novel_genes_df['Gene_Name'].values)

        print(f"Predicted {self.cfg.phenotype} genes: {len(self.predicted_genes)}")
        print(f"Predicted Known {self.cfg.phenotype} genes: {len(self.predicted_known_genes)}")
        print(f"Predicted Novel {self.cfg.phenotype} genes: {len(self.predicted_novel_genes)}")


    def process_gene_proba_predictions(self, top_hits=50, save_to_file=False, top_hits_to_save=1000,
                                       make_plots=True, pos_decision_thres=0.50):

        # Read dictionary text file with gene prediciton probabilities
        proba_df_file = self.cfg.out_root / ('../../benchmarking-for-iterations/gene_proba_predictions/iterations_' + str(self.iterations) + '.' + self.clf_id + '.all_genes.predicted_proba.h5')
        try:
            self.gene_proba_df = pd.read_hdf(proba_df_file, key='df')
            print('Completed reading of ' + 'iterations_' + str(self.iterations) + '.' + self.clf_id + '.all_genes.predicted_proba.h5')
        except Exception as e:
            print(e, '\nFile Not Found: ' + str(proba_df_file) + '.\nSkipping ' + self.clf_id + ' classifier')
            return -1


        # ===== Boxplot for 'All' genes =====
        all_genes_top_hits = self.gene_proba_df.iloc[:, 0:top_hits].columns.values

        # ===== Boxplot for 'Known' genes =====
        #self.known_gene_proba_df = self.gene_proba_df[self.known_genes]


        # Boxplot for 'Unlabelled' genes
        #unlabelled_genes = self.known_genes_df.loc[self.known_genes_df.known_gene == 0, 'Gene_Name'].values
        #self.unlabelled_gene_proba_df = self.gene_proba_df[unlabelled_genes]




if __name__ == '__main__':

    config_file = sys.argv[1] #'../../config.yaml'
    cfg = Config(config_file)

    clf_id = 'ExtraTreesClassifier'

    known_genes_cnt = []
    novel_genes_cnt = []

    iterations = [1, 10, 30, 50, 70, 100, 150, 200]
    # iterations = [10, 30, 50, 70]

    mean_gene_proba_per_iter_df = pd.DataFrame()

    for iter in iterations:
        clf_eval = IterationsBenchmark(cfg, clf_id, iter)

        clf_eval.get_definitive_gene_predictions(pos_ratio_thres=0.99)
        known_genes_cnt.append(len(clf_eval.predicted_known_genes))
        novel_genes_cnt.append(len(clf_eval.predicted_novel_genes))

        clf_eval.process_gene_proba_predictions(top_hits=50)


        print('clf_eval.gene_proba_df.shape:', clf_eval.gene_proba_df.shape)
        cur_gene_proba_means = clf_eval.gene_proba_df.mean(axis=0)
        print(len(cur_gene_proba_means))

        if len(mean_gene_proba_per_iter_df) > 0:
            tmp_df = pd.DataFrame({iter: cur_gene_proba_means})
            mean_gene_proba_per_iter_df = pd.merge(mean_gene_proba_per_iter_df, tmp_df, left_index=True, right_index=True)
            print('Iter:', iter, ' Merged df shape:', mean_gene_proba_per_iter_df.shape)
        else:
            tmp_df = pd.DataFrame({iter: cur_gene_proba_means})
            mean_gene_proba_per_iter_df = tmp_df
        print(mean_gene_proba_per_iter_df.head())

    print(mean_gene_proba_per_iter_df.head())
    iter_cols = mean_gene_proba_per_iter_df.columns.values
    for i in range(mean_gene_proba_per_iter_df.shape[1]):
        for j in range(i+1, mean_gene_proba_per_iter_df.shape[1]):
            if i != j:
               cur_corr = pearsonr(mean_gene_proba_per_iter_df.iloc[:, i], mean_gene_proba_per_iter_df.iloc[:, j])
               print('Iterations:', iter_cols[i], iter_cols[j], ' - Corr:', cur_corr)


    fig, ax = plt.subplots(figsize=(10, 10))
    corr = mean_gene_proba_per_iter_df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                #cmap=sns.light_palette((210, 90, 60), input="husl"),
                cmap=sns.color_palette("ch:3,r=.2,l=.9"),
                square=True,
                annot=True, fmt='.4f')
    ax.xaxis.set_ticks_position('top')
    ax.set_title('Correlation of gene probability predictions\nfor different numbers of stochastic iterations',
                 fontsize=20, y=1.06)
    ax.set_xlabel('Number of stochastic iterations', fontsize=16)
    ax.set_ylabel('Number of stochastic iterations', fontsize=16)

    plt.show()
    ax.get_figure().savefig(str(cfg.out_root / '../../benchmarking-for-iterations/Gene_proba_correlation-Per_Iterations.pdf'),
                            bbox_inches='tight')
    print(corr)
    print(type(corr))


    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Known/Novel genes count per number N of iterations\nin stochastic positive-unlabelled learning', fontsize=20)
    ax.set_xlabel('Number of (partitioning) iterations N', fontsize=16)
    ax.set_ylabel('Gene count', fontsize=16)

    plt.plot(iterations, known_genes_cnt, color='#3182bd', label='Known genes')
    plt.plot(iterations, known_genes_cnt, 'bo', color='#3182bd')

    plt.plot(iterations, novel_genes_cnt, color='#31a354', label='Novel genes')
    plt.plot(iterations, novel_genes_cnt, 'bo', color='#31a354')

    ax.legend(loc='upper right', fontsize='large')
    plt.grid()
    plt.show()
    ax.get_figure().savefig(str(cfg.out_root / '../../benchmarking-for-iterations/Iterations-benchmarking.pdf'), bbox_inches='tight')
