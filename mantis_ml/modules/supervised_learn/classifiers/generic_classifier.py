import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical

from mantis_ml.modules.supervised_learn.core.ml_plot_functions import plot_confusion_matrix

class GenericClassifier:

    def __init__(self, cfg, X_train, y_train, X_test, y_test,
                 train_gene_names, test_gene_names, clf_id, verbose=False):

        self.cfg = cfg
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_gene_names = train_gene_names
        self.test_gene_names = test_gene_names
        self.clf_id = clf_id
        self.verbose = verbose

        self.model = None
        self.y_prob = None
        self.y_pred = None
        self.y_prob = None
        self.train_acc = None
        self.test_acc = None
        self.auc_score = None

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

        self.y_prob = self.model.predict_proba(self.X_test)
        self.y_pred = np.argmax(self.y_prob, axis=1)

        # convert target variables to 2D arrays
        self.y_test = np.array(to_categorical(self.y_test, 2))

        self.conc_pred_df = pd.concat([pd.DataFrame(self.y_test), pd.DataFrame(self.y_prob)], axis=1)
        self.conc_pred_df.columns = ['test_0', 'test_1', 'pred_0', 'pred_1']

    def evaluate_model(self):
        self.train_acc = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        self.test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)

        self.train_acc = round(self.train_acc[1] * 100, 2)
        self.test_acc = round(self.test_acc[1] * 100, 2)

        y_test_roc = np.argmax(self.y_test, axis=1)
        y_pred_roc = self.y_prob[:, 1]
        self.auc_score = roc_auc_score(y_test_roc, y_pred_roc)
        if self.verbose:
            print(f"Training | Test Accuracy: {self.train_acc}% | {self.test_acc}%")
            print(f"AUC: {self.auc_score}")

    def plot_confusion_matrix(self, verbose=True):

        TN, FP, FN, TP = confusion_matrix(np.argmax(self.y_test, axis=1), self.y_pred).ravel()
        accuracy = round(100 * (TP + TN) / (TP + TN + FP + FN), 2)

        if verbose:
            print(classification_report(np.argmax(self.y_test, axis=1), self.y_pred))
            print('Accuracy:', accuracy, '%\n')
            print("TP:", TP)
            print("FN:", FN)
            print("TN:", TN)
            print("FP:", FP)

        confusion = confusion_matrix(np.argmax(self.y_test, axis=1), self.y_pred)

        plot_confusion_matrix(confusion, classes=['unrelated', 'known CKD genes'],
                              title=self.clf_id.replace('Classifier', ''), pdf_filepath=str(self.cfg.superv_figs_out / (self.clf_id + "_confusion_matrix.pdf")))


    def plot_roc_curve(self, make_plot=False):

        y_test_roc = np.argmax(self.y_test, axis=1)
        y_pred_roc = self.y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_test_roc, y_pred_roc)
        roc_auc = roc_auc_score(y_test_roc, y_pred_roc)

        if make_plot:
            f = plt.figure(figsize=(6, 6))
            _ = plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
            _ = plt.plot([0, 1], [0, 1], '--', linewidth=0.5)  # random predictions curve

            _ = plt.xlim([0.0, 1.0])
            _ = plt.ylim([0.0, 1.0])
            _ = plt.title(self.clf_id.replace('Classifier', '') + '\nROC (area = %0.3f)' % roc_auc)
            _ = plt.xlabel('False Positive Rate (1 — Specificity)')
            _ = plt.ylabel('True Positive Rate (Sensitivity)')
            plt.grid(True)
            plt.show()

            f.savefig(str(self.cfg.superv_figs_out / (self.clf_id + "_ROC_curve.pdf")), bbox_inches='tight')

        return fpr, tpr

    def aggregate_predictions(self):

        TP_df = self.conc_pred_df.loc[
            (self.conc_pred_df['pred_1'] >= self.conc_pred_df['pred_0']) & (self.conc_pred_df['test_1'] == 1.0)]
        FP_df = self.conc_pred_df.loc[
            (self.conc_pred_df['pred_1'] >= self.conc_pred_df['pred_0']) & (self.conc_pred_df['test_1'] == 0.0)]

        TN_df = self.conc_pred_df.loc[
            (self.conc_pred_df['pred_0'] > self.conc_pred_df['pred_1']) & (self.conc_pred_df['test_0'] == 1.0)]
        FN_df = self.conc_pred_df.loc[
            (self.conc_pred_df['pred_0'] > self.conc_pred_df['pred_1']) & (self.conc_pred_df['test_0'] == 0.0)]

        self.tp_genes = self.test_gene_names.iloc[TP_df.index]
        self.fp_genes = self.test_gene_names.iloc[FP_df.index]

        self.fn_genes = self.test_gene_names.iloc[FN_df.index]
        self.tn_genes = self.test_gene_names.iloc[TN_df.index]

        if self.verbose:
            self.tp_genes = self.tp_genes.sort_values()
            self.fp_genes = self.fp_genes.sort_values()
            self.fn_genes = self.fn_genes.sort_values()
            self.tn_genes = self.tn_genes.sort_values()

            total_genes = len(self.tn_genes) + len(self.tp_genes) + len(self.fn_genes) + len(self.fp_genes)

            print('* Number of True Positives: **' + str(len(self.tp_genes)) + '**')
            print('* Number of Novel Genes: **' + str(len(self.fp_genes)) + '**')

            print('\n* Number of False Negatives: **' + str(len(self.fn_genes)) + '**')
            print('* Number of True Negatives: **' + str(len(self.tn_genes)) + '**')
            print('* Total num of genes: **' + str(total_genes) + '**')

            print('**List of Novel Genes**: ' + ', '.join(list(self.fp_genes)))
            print('**List of Known Genes**: ' + ', '.join(list(self.tp_genes)) )
