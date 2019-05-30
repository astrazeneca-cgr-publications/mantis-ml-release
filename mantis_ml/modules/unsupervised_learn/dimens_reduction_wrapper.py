import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.use('agg')
import pandas as pd
import os, sys
from mantis_ml.config_class import Config

from mantis_ml.modules.unsupervised_learn.dimens_reduction import DimensionalityReduction
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DimensReductionWrapper(DimensionalityReduction):

    def __init__(self, cfg, data, gene_annot_list, recalc, tsne_perplex=30):
        DimensionalityReduction.__init__(self, cfg)

        self.data = data
        self.gene_annot_list = gene_annot_list
        self.recalc = recalc
        self.tsne_perplex = tsne_perplex

    def run_pca(self, method='PCA'):

        # try:
        # Calculate principal components
        pca, pca_df = self.calc_principal_components(self.data, method=method)
        print(pca_df.head())

        # Plot PCA
        plot_title = "Principal Component Analysis"
        self.plot_embedding_w_labels(pca_df, self.gene_annot_list, 'PC1', 'PC2',
                                plot_title=plot_title, filename_prefix=method, figsize=(12, 12))

        # Store Interactive PCA to .html file
        interactive_pca_df = pca_df.copy()
        interactive_pca_df.rename(columns={'PC1': 'x', 'PC2': 'y'}, inplace=True)
        self.plot_interactive_viz(interactive_pca_df, self.gene_annot_list, method, 1, 0)

        # Make Scree Plot
        if method == 'PCA':
            self.make_scree_plot(pca)

        # except Exception as e:
        #     print('[Exception]:', e)

        # TODO: 3D PCA scatterplot -- https://plot.ly/python/3d-scatter-plots



    def run_tsne(self, data_type='original_data'):

        method = "t-SNE." + data_type + '.perplexity' + str(self.tsne_perplex)
        plot_title = "t-SNE (perplexity=" + str(self.tsne_perplex) + ")"

        X_tsne = None
        total_time = -1.0
        stored_Xtsne_data = str(self.cfg.unsuperv_out / ("tSNE.perplexity" + str(self.tsne_perplex) + "." + data_type + ".tsv"))
        print(stored_Xtsne_data)

        if os.path.exists(stored_Xtsne_data) and not self.recalc:
            X_tsne = pd.read_csv(stored_Xtsne_data, sep = '\t', index_col = False)
        else:
            print('Calculating t-SNE with perplexity:', self.tsne_perplex)
            X_tsne, total_time = self.calc_tsne(self.data, data_type=data_type, perplexity=self.tsne_perplex)

        print(f"[t-SNE] Total time elapsed: {total_time}s")
        print(X_tsne.head())


        print('Plotting t-SNE embedding with selected gene labels...')
        self.plot_embedding_w_labels(X_tsne, self.gene_annot_list, 'd0', 'd1',
                                plot_title=plot_title, filename_prefix=method, figsize=(14, 12))

        interactive_X_tsne = X_tsne.copy()
        interactive_X_tsne.rename(columns={'d0': 'x', 'd1': 'y'}, inplace=True)
        self.plot_interactive_viz(interactive_X_tsne, self.gene_annot_list, method, 1, 0)

        # TODO: complete automated nested-clustering
        # print('Getting clusters (agglomerative) on t-SNE...')
        # agglom_cl, tsne_repr, gene_names = get_clustering_from_tsne(X_tsne, n_clusters=15, perplexity=self.tsne_perplex)
        #
        #
        # filename_prefix = 't-SNE.perplexity' + str(self.tsne_perplex) + '.with_cluster_annotation'
        # plot_embedding_w_clusters(agglom_cl, tsne_repr, gene_list=self.cfg.gene_annot_list,
        #                           gene_names=gene_names,
        #                           filename_prefix=filename_prefix)



    def run_umap(self, data_type='original_data'):

        method = "UMAP." + data_type
        plot_title = "UMAP"

        X_umap = None
        total_time = -1.0
        stored_Xumap_data = str(self.cfg.unsuperv_out / ("UMAP." + data_type + ".tsv"))
        print(stored_Xumap_data)

        if os.path.exists(stored_Xumap_data) and not self.recalc:
            X_umap = pd.read_csv(stored_Xumap_data, sep = '\t', index_col = False)
        else:
            X_umap, total_time = self.calc_umap(self.data, data_type=data_type)

        print(f"[UMAP] Total time elapsed: {total_time}s")
        print(X_umap.head())


        print('Plotting UMAP embedding with selected gene labels...')
        self.plot_embedding_w_labels(X_umap, self.gene_annot_list, 'd0', 'd1',
                                plot_title=plot_title, filename_prefix=method, figsize=(14, 12))

        interactive_X_umap = X_umap.copy()
        interactive_X_umap.rename(columns={'d0': 'x', 'd1': 'y'}, inplace=True)
        self.plot_interactive_viz(interactive_X_umap, self.gene_annot_list, method, 1, 0)




    def run(self):
        
        # > UMAP
        self.run_umap()

        print("\n>> Running Unsupervised analysis...")
        # > PCA
        self.run_pca()

        # > Sparse PCA
        # self.run_pca(method='SparsePCA')

        # > t-SNE
        self.run_tsne()

        print("...Unsupervised analysis complete.")

        # TODO: clustering of t-SNE plot and Pathway Enrichment analysis of CoInterest with PANTHER/IPA


if __name__ == '__main__':

    config_file = sys.argv[1] #'../../config.yaml'
    cfg = Config(config_file)

    # clf_id = 'ExtraTreesClassifier'
    # et_novel_genes = pd.read_csv(str(self.cfg.superv_ranked_pred / (clf_id + '.Novel_genes.Ranked_by_prediction_proba.csv')), header=None, index_col=0)
    # print(et_novel_genes.head())
    # gene_annot_list = et_novel_genes.head(10).index.values

    gene_annot_list = cfg.gene_annot_list
    if len(sys.argv) > 2:
        gene_list_file = sys.argv[2]
        gene_annot_list = pd.read_csv(gene_list_file, header=None)
        gene_annot_list = gene_annot_list.iloc[ :, 0].tolist()

        gene_annot_list = gene_annot_list + cfg.gene_annot_list
    print(gene_annot_list)


    recalc = False # Default: True

    data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')
    dim_reduct_wrapper = DimensReductionWrapper(cfg, data, gene_annot_list, recalc=recalc)
    dim_reduct_wrapper.run()
