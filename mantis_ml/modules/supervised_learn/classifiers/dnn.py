import pandas as pd
import numpy as np
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.utils import to_categorical
from keras.optimizers import Adam

from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.prepare_train_test_sets import PrepareTrainTestSets
from mantis_ml.modules.supervised_learn.classifiers.generic_classifier import GenericClassifier
from mantis_ml.modules.supervised_learn.classifiers.ensemble_lib import ensemble_clf_params



class DnnClassifier(GenericClassifier):

    def __init__(self, cfg, X_train, y_train, X_test, y_test,
                 train_gene_names, test_gene_names, clf_id,
                 regl, hidden_layer_nodes,
                 add_dropout, dropout_ratio, optimizer,
                 epochs, batch_size, verbose, make_plots):

        GenericClassifier.__init__(self, cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, clf_id)

        self.regl = regl
        self.hidden_layer_nodes = hidden_layer_nodes
        self.add_dropout = add_dropout
        self.dropout_ratio = dropout_ratio
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.make_plots = make_plots

        self.model = Sequential()
        self.num_hidden_layers = len(hidden_layer_nodes)
        self.conc_pred_df = None


    def build_model(self):

        for l in range(self.num_hidden_layers):
            # add hidden layer
            self.model.add(Dense(self.hidden_layer_nodes[l], activation='relu',
                                 input_shape=(self.X_train.shape[1],),
                                 kernel_regularizer=regularizers.l2(self.regl)))
            # add dropout layer (if flag is True)
            if self.add_dropout:
                self.model.add(Dropout(self.dropout_ratio))

        self.model.add(Dense(2, activation='softmax'))

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        if self.verbose:
            self.model.summary()

    def train_model(self):
        # do not apply callback without checking
        self.callbacks = [EarlyStopping(monitor='acc', patience=4)]

        self.out = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                                  verbose=self.verbose, validation_split=0.1)  # ,
        # callbacks=self.callbacks)

        self.y_pred = self.model.predict_classes(self.X_test)

        self.y_prob = self.model.predict_proba(self.X_test)

        self.conc_pred_df = pd.concat([pd.DataFrame(self.y_test), pd.DataFrame(self.y_prob)], axis=1)
        self.conc_pred_df.columns = ['test_0', 'test_1', 'pred_0', 'pred_1']

    def plot_training_progression(self):
        epochs = np.array(self.out.__dict__['epoch'])
        acc = np.array(self.out.__dict__['history']['acc'])
        loss = np.array(self.out.__dict__['history']['loss'])
        val_acc = np.array(self.out.__dict__['history']['val_acc'])
        val_loss = np.array(self.out.__dict__['history']['val_loss'])

        fig, ax = plt.subplots(1, 2, figsize=(14, 8))

        # loss
        _ = sns.lineplot(epochs, 100 * loss, color="#fb6a4a", label='training', ax=ax[0])
        _ = sns.lineplot(epochs, 100 * val_loss, label='validation', ax=ax[0])
        _ = ax[0].set_title('Loss')
        _ = ax[0].set_xlabel('epochs')
        _ = ax[0].set_ylabel('% Loss ratio')

        # accuracy
        _ = sns.lineplot(epochs, 100 * acc, color="#fb6a4a", label='training', ax=ax[1])
        _ = sns.lineplot(epochs, 100 * val_acc, label='validation', ax=ax[1])
        _ = ax[1].set_title('Accuracy')
        _ = ax[1].set_xlabel('epochs')
        _ = ax[1].set_ylabel('% Accuracy ratio')

        plt.show()

        fig.savefig(str(self.cfg.superv_figs_out / 'DNN_train_validation-learning_progression.pdf'), format='pdf', bbox_inches='tight')


    def run(self):
        self.build_model()
        self.compile_model()
        self.train_model()
        if self.make_plots:
            self.plot_training_progression()
            self.plot_confusion_matrix()
            self.plot_roc_curve()

        self.evaluate_model()
        self.aggregate_predictions()



if __name__ == '__main__':

    config_file = '../../../config.yaml'
    cfg = Config(config_file)

    set_generator = PrepareTrainTestSets(cfg)

    data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')
    train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(data)


    # select random balanced dataset
    i = 8 #random.randint(0, len(train_dfs))
    print(f"i: {i}")
    train_data = train_dfs[i]
    test_data = test_dfs[i]
    print(f"Training set size: {train_data.shape[0]}")
    print(f"Test set size: {test_data.shape[0]}")

    X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data,
                                                                                                    test_data)

    # convert target variables to 2D arrays
    y_train = np.array(to_categorical(y_train, 2))
    y_test = np.array(to_categorical(y_test, 2))

    # === DNN parameters ===
    dnn_params = ensemble_clf_params['DNN']
    dnn_params['clf_id'] = 'DNN'
    dnn_params['verbose'] = True
    dnn_params['make_plots'] = True
    # ======================

    dnn_model = DnnClassifier(cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, **dnn_params)

    dnn_model.run()
    print(dnn_model.auc_score)
