import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from umap import UMAP
from time import time
import random
import sys

import seaborn as sns
from bokeh.plotting import figure, output_file, save
from bokeh.io import export_svgs, show
from bokeh.models import HoverTool, ColumnDataSource


class DimensionalityReduction:

    def __init__(self, cfg):
        self.cfg = cfg


    def calc_principal_components(self, df, n_comp=20, method='PCA'):
        '''
        Run PCA and Sparse PCA on feature table
        :param df: 
        :return: 
        '''
        print(">> Running " + method + "...")
        if df.shape[1] <= n_comp:
            n_comp = df.shape[1] - 1

        tmp_drop_cols = ['Gene_Name', self.cfg.Y]
        X = df.drop(tmp_drop_cols, axis=1)
        pca_data = X.copy()

        pca = None
        if method == 'SparsePCA':
            pca = SparsePCA(n_components=n_comp)
        else:
            pca = PCA(n_components=n_comp)
        principal_components = pca.fit_transform(pca_data)

        columns = []
        for i in range(1, n_comp+1):
            columns.append('PC' + str(i))

        pca_df = pd.DataFrame(data = principal_components, columns = columns)
        pca_df = pd.concat([pca_df, df[tmp_drop_cols]], axis=1)

        filepath = str(self.cfg.unsuperv_out / (method + ".table.tsv"))
        pca_df.to_csv(filepath, sep='\t', index=None)

        return pca, pca_df


    def make_scree_plot(self, pca, method='PCA'):
        var = pca.explained_variance_ratio_
        print(type(pca))
        print(pca)
        print(pca.n_components_)
        n_comp_to_show = pca.n_components_

        # cum_var = np.cumsum(np.round(var, decimals=4) * 100)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.bar(range(1, n_comp_to_show + 1), var * 100)
        plt.title(method + ' - Scree Plot')
        plt.xlabel('Principal Components')
        plt.ylabel('% Variance explained')
        plt.xticks(range(1, n_comp_to_show+1), ['PC' + str(i) for i in range(1, n_comp_to_show+1)], fontsize=8)
        # plt.show()

        plot_filename = method + "_Scree_plot.pdf"
        fig.savefig(str(self.cfg.unsuperv_figs_out / plot_filename), bbox_inches='tight')


    def calc_umap(self, df, n_neighbors=5, min_dist=0.3, metric='correlation', data_type='original_data'):

        print(">> Running UMAP from " + data_type + "...")
        tmp_drop_cols = ['Gene_Name', self.cfg.Y]
        X = df.drop(tmp_drop_cols, axis=1)

        umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        t0 = time()
        X_umap = umap.fit_transform(X)
        total_time = time() - t0

        X_umap = pd.DataFrame(X_umap)
        X_umap.columns = [('d' + str(c)) for c in X_umap.columns.values]
        #print(X_umap)

        X_umap = pd.concat([X_umap, df[tmp_drop_cols]], axis=1)

        filepath = str(self.cfg.unsuperv_out / ("UMAP" + data_type + ".tsv"))
        X_umap.to_csv(filepath, sep='\t', index=None)

        return X_umap, total_time




    def calc_tsne(self, df, n_comp=2, data_type='original_data', perplexity=30):
        '''
        Calculate t-SNE
        :param df: 
        :param n_comp: 
        :param data_type: table used for t-SNE calculations - 'original_data' or 'principal_components' 
        :return: 
        '''

        print(">> Running t-SNE from " + data_type + "...")
        tmp_drop_cols = ['Gene_Name', self.cfg.Y]
        X = df.drop(tmp_drop_cols, axis=1)

        tsne = TSNE(n_comp, init='pca', random_state=0, perplexity=perplexity)
        t0 = time()
        X_tsne = tsne.fit_transform(X)
        total_time = time() - t0


        X_tsne = pd.DataFrame(X_tsne)
        X_tsne.columns = [('d' + str(c)) for c in X_tsne.columns.values]
        #print(X_tsne)

        X_tsne = pd.concat([X_tsne, df[tmp_drop_cols]], axis=1)

        filepath = str(self.cfg.unsuperv_out / ("tSNE.perplexity" + str(perplexity) + "." + data_type + ".tsv"))
        X_tsne.to_csv(filepath, sep='\t', index=None)

        return X_tsne, total_time


    def get_clustering_from_tsne(self, X_tsne, n_clusters=15, perplexity=30):

        gene_names = X_tsne['Gene_Name']
        known_genes = X_tsne[self.cfg.Y]

        tsne_repr = X_tsne.drop([self.cfg.Y, 'Gene_Name'], axis=1)

        agglom_cl = AgglomerativeClustering(n_clusters)
        agglom_cl.fit(tsne_repr)

        tsne_repr.columns = ['x', 'y']
        tsne_repr['cluster'] = agglom_cl.labels_
        tsne_repr['Gene_Name'] = gene_names
        tsne_repr['known_CKD_gene'] = known_genes

        return agglom_cl, tsne_repr, gene_names


    def plot_embedding_w_clusters(self, agglom_cl, tsne_repr, gene_list=[], gene_names=None, filename_prefix='embedding_w_clusters', figsize=(16, 16)):

        plt.rc('font', size=14)
        sns.set_style('white')

        # define a custom palette
        palette = sns.color_palette("Paired") + sns.color_palette("Set2")
        palette = palette[:agglom_cl.n_clusters]

        fig, ax = plt.subplots(figsize=figsize)
        _ = plt.title('t-SNE plots with highlighted k-means clusters (k=15)')

        for i in range(agglom_cl.n_clusters):
            _ = ax.scatter(x=tsne_repr.loc[tsne_repr.cluster == i, 'x'],
                           y=tsne_repr.loc[tsne_repr.cluster == i, 'y'],
                           color=palette[i], label=i, s=40, marker='.')
        lgnd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Clusters', fancybox=True)
        for handle in lgnd.legendHandles:
            handle.set_sizes([500])

        compon1 = list(tsne_repr.loc[:, 'x'])
        compon2 = list(tsne_repr.loc[:, 'y'])

        for i, gene in enumerate(gene_list):  # enumerate(list(gene_names)):
            idx = gene_names[gene_names == gene].index[0]
            _ = ax.text(compon1[idx], compon2[idx] + random.randint(1, 4), gene)

        fig.savefig(str(self.cfg.unsuperv_figs_out / (filename_prefix + '.pdf')), bbox_inches='tight')



    def plot_embedding_w_labels(self, df, highlighted_genes, x, y, plot_title, filename_prefix, figsize=(10, 10)):
        '''
        Plot a (static) dimensionality reduction embedding (e.g. PCA, t-SNE)
        with label annotation for selected data points
        '''

        gene_names = df['Gene_Name']

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(plot_title, fontsize=20)

        targets = [0, 1]
        colors = ['#bdbdbd', '#ef3b2c']

        for target, color in zip(targets, colors):
            indicesToKeep = df[self.cfg.Y] == target
            ax.scatter(df.loc[indicesToKeep, x],
                           df.loc[indicesToKeep, y],
                           c=color,
                           s=20)

        plt.xlabel("Dimension-1")
        plt.ylabel("Dimension-2")
        ax.legend(targets, loc=2)

        compon1 = list(df.loc[:, x])
        compon2 = list(df.loc[:, y])

        for i, gene in enumerate(highlighted_genes):
            try:
                idx = gene_names[gene_names == gene].index[0]
                ax.annotate(gene, (compon1[idx], compon2[idx]))
            except Exception as e:
                print('[Warning]:', gene, ' not found in gene list')
        plot_filename = filename_prefix + "_plot.pdf"
        fig.savefig(str(self.cfg.unsuperv_figs_out / plot_filename), bbox_inches='tight')


    def plot_interactive_viz(self, data, highlighted_genes, method, pos_label, neg_label, show_plot=False, save_plot=False):
        '''
        Plot an interactive dimensionality reduction embedding (e.g. PCA, t-SNE)
        with label annotation for selected data points
        '''

        # Highlight genes of interest
        data['colors'] = data.known_gene.copy()
        color_mapping = {pos_label: '#ef3b2c', neg_label: '#bdbdbd'}
        data = data.replace({'colors': color_mapping})
        data = data.sort_values(by=[self.cfg.Y], ascending=True)

        known_genes_highlight_color = '#31a354'
        data.loc[data['Gene_Name'] == 'PKD1', 'colors'] = known_genes_highlight_color
        data.loc[data['Gene_Name'] == 'PKD2', 'colors'] = known_genes_highlight_color

        selected_gene_rows = data.loc[data['Gene_Name'].isin(highlighted_genes), :]
        data = data[~data.Gene_Name.isin(highlighted_genes)]
        data = pd.concat([data, selected_gene_rows], axis=0)
        data.loc[data['Gene_Name'].isin(highlighted_genes), 'colors'] = '#252525'

        data['annotation'] = data.known_gene.copy()
        data.loc[data.annotation == pos_label, 'annotation'] = 'Yes'
        data.loc[data.annotation == neg_label, 'annotation'] = 'No'

        # Plot
        source = ColumnDataSource(dict(
            x=data['x'],
            y=data['y'],
            color=data['colors'],
            content=data['Gene_Name'],
            annot=data['annotation'],
        ))

        interact_viz = figure(plot_width=900, plot_height=900,
                              title=method, tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                              x_axis_type=None, y_axis_type=None, min_border=1)

        interact_viz.scatter(x='x', y='y',
                             source=source,
                             color='color',
                             alpha=0.8, size=10,
                             legend=method)

        # hover tools
        hover = interact_viz.select(dict(type=HoverTool))
        hover.tooltips = [("gene", "@content")]
        interact_viz.legend.location = "top_left"

        plot_filename = method + "_interactive_plot.html"
        output_file(str(self.cfg.unsuperv_figs_out / plot_filename))
        save(interact_viz)

        if show_plot:
            show(interact_viz)

        if save_plot:
            interact_viz.output_backend = "svg"
            plot_filename = method + '_interactive_plot.svg'
            export_svgs(interact_viz, filename=(self.cfg.unsuperv_figs_out / plot_filename))
