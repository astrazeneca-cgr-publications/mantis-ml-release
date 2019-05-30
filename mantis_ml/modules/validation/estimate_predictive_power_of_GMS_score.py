import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from mantis_ml.config_class import Config



config_file = '../../input_configs/validation.Generic_config.yaml' #sys.argv[1]
cfg = Config(config_file)

# *************************************************************
# ********* GMS: Generic Mantis-ml Score *********
#
#            To be used with the Generic classifier
# *************************************************************

color_palette = {'OMIM_MGI_union': 'black', 'seed': '#33a02c', 'hidden_seed': '#fb9a99', 'unlabelled_sample': '#1f78b4', 'unlabelled': '#969696',
                'omim_recessive': '#33a02c', 'omim_haploinsufficient': '#b2df8a', 'omim_dom_neg': '#1f78b4', 'omim_de_novo': '#a6cee3',
                'omim_de_novo_and_haploinsuf': '#e31a1c', 'mgi_lethal_orthologs': '#fdbf6f', 'mgi_seizure_orthologs':'#cab2d6'}

merged_df = pd.read_csv(cfg.superv_out / 'Seed_Hidden_Unlabelled_proba_means.csv', index_col=0)
merged_df['mantis_ml_perc'] = 100 * merged_df['Proba'].rank(pct=True)
merged_df.head()

hidden_seed_genes = pd.read_csv(cfg.out_data_dir / 'hidden_seed_genes.txt', header=None)
hidden_seed_genes = list(hidden_seed_genes.iloc[:, 0].values)
print('Hidden seed genes:', len(hidden_seed_genes))


def plot_roc_curve(fpr, tpr, roc_auc, gene_class):
    if gene_class in color_palette:
        plot_color = color_palette[gene_class]
    else:
        plot_color = 'black'
    plt.plot(fpr, tpr, color=plot_color,
             label=gene_class + ' (AUC = %0.4f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver operating characteristic', fontsize=16)

    plt.legend(loc="lower right", fontsize=14)



# ============ Find optimal cutoff for logistic regression with mantis-ml scores ============
seed_unlabelled_df = merged_df.loc[merged_df.Class != 'hidden_seed']
hidden_unlabelled_df = merged_df.loc[merged_df.Class != 'seed']
lr = LogisticRegression(C=1e9, solver='lbfgs')


def run_logist_regression(df, model, gene_class):
    df = df.copy()
    df['Class'] = df['Class'].map({'seed': 1, 'hidden_seed': 1, 'unlabelled': 0})
    df = df.sample(frac=1)
    print(df.head())

    feature_cols = ['Proba']
    X = df[feature_cols]
    y = df['Class']
    print('X:', X.shape)
    print('y:', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print('y_train:', y_train.shape)
    print(X_train.head())
    print(X_test.head())
    print(type(X_train))

    # Train Log. Regr. classifier on all training data
    lr.fit(X_train, y_train)
    # Predict on test data
    y_test_proba = model.predict_proba(X_test)[:, 1]

    print('y_test:', y_test_proba.shape)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    # ===================== BETA =====================
    # -- Find optimacl cutoff based on ROC curve
    #     i = np.arange(len(tpr))
    #     roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(thresholds, index=i)})
    #     roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    #     print(list(roc_t['threshold']))

    #     optimal_idx = np.argmax(tpr - fpr)
    #     optimal_threshold = thresholds[optimal_idx]
    #     print('Optimal threshold:', optimal_threshold)
    # ================================================

    # CV accuracy
    # predicted = cross_val_predict(lr, X_train, y_train, cv=10)
    # print("Cross-Validation Accuracy is", accuracy_score(y_train, predicted) * 100)

    plot_roc_curve(fpr, tpr, roc_auc, gene_class)


fig, ax = plt.subplots(figsize=(10, 10))
run_logist_regression(seed_unlabelled_df, lr, 'seed')
run_logist_regression(hidden_unlabelled_df, lr, 'hidden_seed')
plt.show()
fig.savefig(cfg.superv_figs_out / 'GMS_score_ROC_predictive_power.pdf', bbox_inches='tight')



# =============== Read data from RVIS 2013 paper ===============
s1_data = pd.read_csv(cfg.data_dir / 'rvis_plosgen_2013/rvis_dataset_S1.csv', sep='\t')
s1_data.fillna(0, inplace=True)
# print(s1_data.head())

# Check performance only on Hidden genes
s1_data = s1_data.loc[ s1_data['OMIM disease genes'].isin(hidden_seed_genes), :]
# print(s1_data.shape)


omim_recessive = s1_data.loc[s1_data['OMIM Recessive'] != 0, 'OMIM disease genes'].values.tolist()
print('OMIM Recessive:', len(omim_recessive))

omim_haploinsufficient = s1_data.loc[s1_data['OMIM Haploinsufficiency'] != 0, 'OMIM disease genes'].values.tolist()
print('OMIM Haploinsufficiency:', len(omim_haploinsufficient))

omim_dom_neg = s1_data.loc[s1_data['OMIM Dominant Negative'] != 0, 'OMIM disease genes'].values.tolist()
print('OMIM Dominant Negative:', len(omim_dom_neg))

omim_de_novo = s1_data.loc[s1_data['OMIM de novo'] != 0, 'OMIM disease genes'].values.tolist()
print('OMIM de novo:', len(omim_de_novo))

omim_de_novo_and_haploinsuf = s1_data.loc[s1_data['OMIM de novo & Haploinsuficciency'] != 0, 'OMIM disease genes'].values.tolist()
print('OMIM de novo & Haploinsuficciency:', len(omim_de_novo_and_haploinsuf))

mgi_lethal_orthologs = s1_data.loc[s1_data['MGI Lethality orthologs'] != 0, 'OMIM disease genes'].values.tolist()
print('MGI Lethality orthologs:', len(mgi_lethal_orthologs))

mgi_seizure_orthologs = s1_data.loc[s1_data['MGI Seizure orthologs'] != 0, 'OMIM disease genes'].values.tolist()
print('MGI Seizure orthologs:', len(mgi_seizure_orthologs))


omim_mgi_union = list(set(omim_recessive + omim_haploinsufficient + omim_dom_neg + omim_de_novo + omim_de_novo_and_haploinsuf + mgi_lethal_orthologs + mgi_seizure_orthologs))
print('OMIM - MGI union:', len(omim_mgi_union))




s2_data = pd.read_csv(cfg.data_dir / 'rvis_plosgen_2013/rvis_dataset_S2.csv')
s2_data.head()

non_omim_mgi_genes = list(set(s2_data['HGNC gene']) - set(omim_mgi_union))
print('Non OMIM - MGI:', len(non_omim_mgi_genes))


union_of_genes = s1_data['OMIM disease genes'].tolist() + non_omim_mgi_genes
print('Union of OMIM and non-OMIM genes:', len(union_of_genes))




full_feature_df = pd.read_csv(cfg.compiled_data_dir / 'complete_feature_table.tsv', sep='\t')
full_feature_df.head()

intolerance_scores = ['GnomAD_mis_z', 'GnomAD_pLI', 'ExAC_cnv.score', 'LoF_FDR_ExAC', 'RVIS_ExACv2']
full_feature_df = full_feature_df[['Gene_Name'] + intolerance_scores]
full_feature_df.head()
full_feature_df.shape

full_feature_df = full_feature_df.loc[full_feature_df.Gene_Name.isin(union_of_genes), :]
full_feature_df.shape




#  =============== RVIS vs mantis-ml correlation of percentile scores  ===============
mantis_ml_score = 'mantis_ml_perc' #Proba
all_mantis_ml_proba = merged_df[['Gene_Name', mantis_ml_score]]

all_mantis_ml_proba.shape
all_mantis_ml_proba = all_mantis_ml_proba.loc[all_mantis_ml_proba.Gene_Name.isin(union_of_genes), :]
all_mantis_ml_proba.head()
all_mantis_ml_proba.shape



intol_scores_df = pd.merge(full_feature_df, all_mantis_ml_proba, how='inner', left_on='Gene_Name', right_on='Gene_Name')


intol_score_mantis_corr = pearsonr(intol_scores_df['RVIS_ExACv2'], intol_scores_df[mantis_ml_score] )
print(intol_score_mantis_corr)
intol_scores_df.head()


fig, ax = plt.subplots(figsize=(9, 9))
plt.scatter(intol_scores_df['RVIS_ExACv2'], intol_scores_df[mantis_ml_score], s=1)
plt.xlabel('intol_score percentile scores')
plt.ylabel('mantis-ml percentile scores')
plt.title('intol_score vs mantis-ml correlation:' + str(intol_score_mantis_corr[0]))
# plt.show()
fig.savefig(cfg.superv_figs_out / 'intol_score_vs_mantis-ml_scatterplot.pdf', bbox_inches='tight')



def get_predictive_power_for_intol_score(intol_score):
    
    lr = LogisticRegression(C=1e9, solver='lbfgs')

    def run_logist_regression(intol_score, positive_genes, negative_genes, model, gene_class):
        df = intol_scores_df[['Gene_Name', intol_score]]

        
        pos_df = intol_scores_df.loc[intol_scores_df.Gene_Name.isin(positive_genes)].copy()
        pos_df['Class'] = gene_class
        print(gene_class + ':', pos_df.shape[0])

        neg_df = intol_scores_df.loc[intol_scores_df.Gene_Name.isin(negative_genes)].copy()
        neg_df['Class'] = 'non_OMIM_MGI'
        #print('Negative set:', neg_df.shape)

        tmp_merged_df = pd.concat([pos_df, neg_df], axis=0)
        tmp_merged_df = tmp_merged_df[['Class', intol_score]]
        #print(tmp_merged_df.head())

        tmp_merged_df['Class'] = tmp_merged_df['Class'].map({'omim_recessive': 1,
                                                             'omim_haploinsufficient': 1,
                                                             'omim_dom_neg': 1,
                                                             'omim_de_novo': 1,
                                                             'omim_de_novo_and_haploinsuf': 1,
                                                             'mgi_lethal_orthologs': 1,
                                                             'mgi_seizure_orthologs': 1,
                                                             'OMIM_MGI_union': 1,
                                                             'non_OMIM_MGI': 0})
        df = df.sample(frac=1)

        feature_cols = [intol_score]
        X = tmp_merged_df[feature_cols]
        y = tmp_merged_df['Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Train Log. Regr. classifier on all training data
        lr.fit(X_train, y_train)
        # Predict on test data
        y_test_proba = model.predict_proba(X_test)[:, 1]
        #print('y_test:', y_test_proba.shape)

        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
        roc_auc = auc(fpr, tpr)
        #print("Area under the ROC curve : %f" % roc_auc)


        plot_roc_curve(fpr, tpr, roc_auc, gene_class)


    fig, ax = plt.subplots(figsize=(10, 10))

    
    run_logist_regression(intol_score, omim_mgi_union, non_omim_mgi_genes, lr, 'OMIM_MGI_union')
    run_logist_regression(intol_score, omim_recessive, non_omim_mgi_genes, lr, 'omim_recessive')
    run_logist_regression(intol_score, omim_haploinsufficient, non_omim_mgi_genes, lr, 'omim_haploinsufficient')
    run_logist_regression(intol_score, omim_dom_neg, non_omim_mgi_genes, lr, 'omim_dom_neg')
    run_logist_regression(intol_score, omim_de_novo, non_omim_mgi_genes, lr, 'omim_de_novo')
    run_logist_regression(intol_score, omim_de_novo_and_haploinsuf, non_omim_mgi_genes, lr, 'omim_de_novo_and_haploinsuf')
    run_logist_regression(intol_score, mgi_lethal_orthologs, non_omim_mgi_genes, lr, 'mgi_lethal_orthologs')
    run_logist_regression(intol_score, mgi_seizure_orthologs, non_omim_mgi_genes, lr, 'mgi_seizure_orthologs')
    plt.title('ROC: ' + intol_score)

    plt.show()
    fig.savefig(cfg.superv_figs_out / (intol_score + '_predictive_power.pdf'), bbox_inches='tight')
    
    
get_predictive_power_for_intol_score('RVIS_ExACv2')

get_predictive_power_for_intol_score('mantis_ml_perc')

get_predictive_power_for_intol_score('GnomAD_mis_z')

get_predictive_power_for_intol_score('GnomAD_pLI')

get_predictive_power_for_intol_score('ExAC_cnv.score')

get_predictive_power_for_intol_score('LoF_FDR_ExAC')