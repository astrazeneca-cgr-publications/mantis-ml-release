import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
import pickle
from palettable.colorbrewer.sequential import Greens_9
from palettable.colorbrewer.qualitative import Paired_12
from matplotlib.patches import Patch
from scipy.stats.stats import pearsonr
import seaborn as sns
import warnings
import matplotlib.cbook 
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

from mantis_ml.config_class import Config
from mantis_ml.modules.post_processing.evaluate_classifier_predictions import ClassifierEvaluator
from mantis_ml.modules.post_processing.process_classifier_results import ProcessClassifierResults
import mantis_ml.modules.post_processing.pyupset as pyu

# classifiers = ['ExtraTreesClassifier', 'DNN', 'RandomForestClassifier', 'SVC', 'XGBoost', 'GradientBoostingClassifier'] #, 'Stacking_DNN']
# classifiers = ['ExtraTreesClassifier', 'SVC', 'Stacking_DNN'] # For CKD
# classifiers = ['ExtraTreesClassifier', 'DNN', 'RandomForestClassifier', 'SVC', 'XGBoost', 'GradientBoostingClassifier', 'Stacking_DNN'] # For Epilepsy
# classifiers = ['ExtraTreesClassifier'] #, 'DNN', 'RandomForestClassifier'] #, 'Stacking_DNN'] # For ALS



class MergePredictionsFromClassifiers:

    def __init__(self, cfg, show_plots=False):
        self.cfg = cfg
        self.show_plots = show_plots

    def get_consensus_of_gene_predictions(self, all_clf, genes_attr='predicted_known_genes'):

        all_genes = dict()

        # *** Number of classifiers that need to have predicted a gene...
        # ... for it to be considered in final predictions ***
        num_clf_thres = (len(self.cfg.classifiers) - 1)

        for clf_id in all_clf:
            if clf_id not in self.cfg.classifiers:
                continue
            print('\n' + clf_id)

            cur_clf = all_clf[clf_id]

            for ng in getattr(cur_clf, genes_attr):
                all_genes[ng] = all_genes.get(ng, 0) + 1

        consensus_pred_genes = [ng for ng in all_genes if all_genes[ng] >= num_clf_thres]
        with open(str(self.cfg.superv_pred / ('All_classifiers.' + genes_attr + '.txt')), 'w') as f:
            for gene in consensus_pred_genes:
                f.write("%s\n" % gene)

        return consensus_pred_genes


    def get_definitive_gene_predictions_overlap(self, all_clf, genes_attr='predicted_known_genes', gene_class='Known'):

        # Limit intersection plots on top 4 classifiers with best AUC performance during benchmarking
        top_classifiers = ['ExtraTreesClassifier', 'RandomForestClassifier', 'XGBoost', 'SVC', 'DNN', 'GradientBoostingClassifier', 'Stacking']

        all_genes_dict = dict()
        for clf_id, clf_eval in all_clf.items():
            if clf_id not in top_classifiers:
                continue

            clf_id = clf_id.replace('Classifier', '')
            cur_genes = getattr(clf_eval, genes_attr)

            all_genes_dict[clf_id] = pd.DataFrame({(gene_class +'_Genes'): cur_genes})

        if len(all_genes_dict.keys()) == 1:
            print("[Warning] 'get_definitive_gene_predictions_overlap' not applicable -- single classifier results provided")
            return 0


        inters_sizes = {'Known': 5, 'Novel': 50}
        pyu.plot(all_genes_dict,
                 sort_by='degree',
                 inters_size_bounds=(inters_sizes[gene_class], np.inf))

        # plt.show()
        cur_fig = matplotlib.pyplot.gcf()
        cur_fig.savefig(str(self.cfg.superv_figs_out / (gene_class + '_genes_Intersections_between_classifiers.pdf')), bbox_inches='tight')


    # TODO:
    def beta_get_gene_predictions_based_on_thres(self, all_clf, thres=0.5, genes_attr='known_gene_proba_means', gene_class='Known'):

        top_classifiers = ['ExtraTreesClassifier', 'RandomForestClassifier', 'XGBoost', 'SVC', 'DNN', 'GradientBoostingClassifier']

        all_genes_dict = dict()
        for clf_id, clf_eval in all_clf.items():
            if clf_id not in top_classifiers:
                continue
            clf_id = clf_id.replace('Classifier', '')
            print('Classifier:', clf_id)

            cur_genes = getattr(clf_eval, genes_attr)
            cur_genes = cur_genes.loc[cur_genes > thres]
            print(gene_class + ' ' + str(len(cur_genes)))

            all_genes_dict[clf_id] = pd.DataFrame({(gene_class + '_Genes'): cur_genes.index.values})


        inters_sizes = {'Known': 5, 'Novel': 50}
        pyu.plot(all_genes_dict,
                 sort_by='degree',
                 inters_size_bounds=(inters_sizes[gene_class], np.inf))
        # plt.show()
        cur_fig = matplotlib.pyplot.gcf()
        cur_fig.savefig(str(self.cfg.superv_figs_out / (gene_class + '_genes.PredThres' + str(thres) + '.Intersections_between_classifiers.pdf')), bbox_inches='tight')



    def merge_proba_from_all_clf(self):

        merged_clf_proba_df = pd.DataFrame()
        for clf_id in self.cfg.classifiers:

            proba_df_file = self.cfg.superv_proba_pred / (clf_id + '.all_genes.predicted_proba.h5')
            gene_proba_df = pd.read_hdf(proba_df_file, key='df')
            gene_proba_df.sort_index(axis=1, inplace=True)
            #print(gene_proba_df.head())

            if len(merged_clf_proba_df) > 0:
                merged_clf_proba_df = pd.concat([merged_clf_proba_df, gene_proba_df], axis=0, sort=True)
            else:
                merged_clf_proba_df = gene_proba_df

            #print(merged_clf_proba_df.shape)

            merged_clf_proba_df = merged_clf_proba_df.reindex(merged_clf_proba_df.mean().sort_values(ascending=False).index, axis=1)
            merged_clf_proba_df.to_hdf(self.cfg.superv_proba_pred / 'AllClassifiers.Merged.all_genes.predicted_proba.h5', key='df')

        return merged_clf_proba_df



    def merge_mean_proba_from_all_clf(self):

        mean_gene_proba_df_per_clf = pd.DataFrame()
        for clf_id in self.cfg.classifiers:

            proba_df_file = self.cfg.superv_proba_pred / (clf_id + '.all_genes.predicted_proba.h5')
            gene_proba_df = pd.read_hdf(proba_df_file, key='df')
            gene_proba_df.sort_index(axis=1, inplace=True)
            #print(gene_proba_df.shape)

            gene_mean_proba_series = gene_proba_df.mean(axis=0)
            #print(gene_mean_proba_series.shape)
            #print(gene_mean_proba_series.head())


            if len(mean_gene_proba_df_per_clf) > 0:
                tmp_df = pd.DataFrame({clf_id: gene_mean_proba_series})

                mean_gene_proba_df_per_clf = pd.merge(mean_gene_proba_df_per_clf, tmp_df, left_index=True, right_index=True)
                print('Classifier:', clf_id, ' Merged df shape:', mean_gene_proba_df_per_clf.shape)
            else:
                tmp_df = pd.DataFrame({clf_id: gene_mean_proba_series})
                mean_gene_proba_df_per_clf = tmp_df

            #print(mean_gene_proba_df_per_clf.head())

        mean_gene_proba_df_per_clf.columns = mean_gene_proba_df_per_clf.columns.str.replace('Classifier', '')
        mean_gene_proba_df_per_clf.index = mean_gene_proba_df_per_clf.index.str.replace('Classifier', '')

        fig, ax = plt.subplots(figsize=(10, 10))

        corr = mean_gene_proba_df_per_clf.corr()

        sns.heatmap(corr,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    #cmap=sns.cubehelix_palette(rot=-.4, as_cmap=True),
                    cmap=sns.color_palette("ch:2,r=.2,l=.9"),
                    square=True,
                    annot=True, fmt='.4f',
                    vmin=0.7, vmax=1)
        ax.xaxis.set_ticks_position('top')
        ax.set_title('Correlation of gene probability predictions\nacross different classifiers',
                     fontsize=20, y=1.06)
        ax.set_xlabel('Classifiers', fontsize=16)
        ax.set_ylabel('Classifiers', fontsize=16)

        # plt.show()
        ax.get_figure().savefig(
            str(self.cfg.benchmark_out / 'Gene_mean_proba_correlation-Per_Classifier.pdf'),
            bbox_inches='tight')

        return mean_gene_proba_df_per_clf


    def run(self):

        self.merge_mean_proba_from_all_clf()
        merged_clf_proba_df = self.merge_proba_from_all_clf()
        #print(merged_clf_proba_df.head())

        top_hits = 50
        top_genes_from_merged_clf = merged_clf_proba_df.columns.values[:top_hits]

        clf_eval = ClassifierEvaluator(self.cfg, 'Gene_Consensus')
        clf_eval.gene_proba_df = merged_clf_proba_df
        clf_eval.gene_ranking_boxplot(top_genes_from_merged_clf, 'Merged_classifiers-All')

        # ============================================================================================

        all_clf_filepath = str(self.cfg.superv_out / 'all_clf.pkl')
        if os.path.exists(all_clf_filepath):
            print("Reading all_clf.pkl")
            with open(all_clf_filepath, 'rb') as input:
                all_clf = pickle.load(input)
        else:
            sys.exit("[Error] -- 'merge_predictions_from_classifiers.py': all_clf.pkl file not found.")

        aggr_res = ProcessClassifierResults(self.cfg, show_plots=False)

        
        consensus_known_genes = self.get_consensus_of_gene_predictions(all_clf, genes_attr='predicted_known_genes')
        consensus_novel_genes = self.get_consensus_of_gene_predictions(all_clf, genes_attr='predicted_novel_genes')

        print('consensus_known_genes:', len(consensus_known_genes))
        print('consensus_novel_genes:', len(consensus_novel_genes))
        #print(consensus_known_genes[:10])

        # Merge results from all classifiers
        merged_clf_proba_df = self.merge_proba_from_all_clf()
        merged_clf_eval = aggr_res.process_merged_clf_proba()
        all_clf['AllClassifiers.Merged'] = merged_clf_eval

        known_merged_clf_proba_df = merged_clf_proba_df[consensus_known_genes]
        known_merged_clf_proba_df = known_merged_clf_proba_df.reindex(
            known_merged_clf_proba_df.mean().sort_values(ascending=False).index, axis=1)
        print('known_merged_clf_proba_df:', known_merged_clf_proba_df.shape)
        #print(known_merged_clf_proba_df.head(20))

        novel_merged_clf_proba_df = merged_clf_proba_df[consensus_novel_genes]
        novel_merged_clf_proba_df = novel_merged_clf_proba_df.reindex(
            novel_merged_clf_proba_df.mean().sort_values(ascending=False).index, axis=1)
        print('novel_merged_clf_proba_df:', novel_merged_clf_proba_df.shape)
        #print(novel_merged_clf_proba_df.head(20))
        # TODO: plot top consensus known and novel genes -- use 'gene_ranking_boxplot' from ClassifierEvaluator class


        # TODO: fix BETA
        # self.beta_get_gene_predictions_based_on_thres(all_clf, thres=0.6)
        # self.beta_get_gene_predictions_based_on_thres(all_clf, thres=0.6, genes_attr='unlabbeled_gene_proba_means', gene_class='Novel')

        self.get_definitive_gene_predictions_overlap(all_clf)
        self.get_definitive_gene_predictions_overlap(all_clf, genes_attr='predicted_novel_genes', gene_class='Novel')
     
        print(all_clf.keys())

        if 'AllClassifiers.Merged' in all_clf:
            del all_clf['AllClassifiers.Merged']

        #aggr_res.plot_gene_counts_per_clf(all_clf)
        aggr_res.get_density_and_cdf_plots(all_clf, 'known_gene_proba_means', gene_class='Known')
        aggr_res.get_density_and_cdf_plots(all_clf, 'gene_proba_means', gene_class='All')
        aggr_res.get_density_and_cdf_plots(all_clf, 'unlabbeled_gene_proba_means', gene_class='Unlabelled')


if __name__ == '__main__':

    config_file = sys.argv[1] #'../../config.yaml'
    cfg = Config(config_file)

    merger = MergePredictionsFromClassifiers(cfg)
    merger.run()
