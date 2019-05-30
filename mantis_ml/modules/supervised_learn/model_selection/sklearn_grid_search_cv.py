from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import random
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb

from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.prepare_train_test_sets import PrepareTrainTestSets
from mantis_ml.modules.supervised_learn.classifiers.ensemble_lib import ensemble_clf_params


def run_grid_search(model, X_train, y_train, param_grid, scoring='roc_auc', cv=10):

    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring=scoring, cv=cv)

    grid.fit(X_train, y_train)

    # Summarize the results of the grid search
    print('- Best AUC:', grid.best_score_)
    print("- Best parameters:")
    for param in param_grid.keys():
        print("'" + param + "': " + str(getattr(grid.best_estimator_, param)) + ",")


def fine_tune_tree_based_models(X_train, y_train, clf_id):

    model = None
    if clf_id == 'ExtraTreesClassifier':
        model = ExtraTreesClassifier(n_estimators=100)
    elif clf_id == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=100)
    elif clf_id == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(n_estimators=100)

    n_estimators = [100, 200, 300, 400, 500]
    max_features = ['auto', 'sqrt', 'log2', None]
    max_depth = [5, 10, 15, 20]
    min_samples_leaf = range(2, 6)
    min_samples_split = range(2, 6)

    if small_search_space:
        n_estimators = [100, 200]
        max_features = ['auto']
        max_depth = [5]
        min_samples_leaf = [2]
        min_samples_split = [2]

    param_grid = dict(n_estimators=n_estimators,
                      max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf,
                      min_samples_split=min_samples_split,
                      max_features=max_features)

    print('\n> ' + clf_id + ':')
    run_grid_search(model, X_train, y_train, param_grid)


def fine_tune_svc(X_train, y_train):

    model = SVC(gamma='auto')

    C = [0.01, 0.025, 0.1, 0.25, 1]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    shrinking = [True, False]
    probability = [True, False]

    if small_search_space:
        C = [0.01, 0.1, 1]
        kernel = ['linear','rbf']
        shrinking = [True]
        probability = [True]

    param_grid = dict(C=C,
                      kernel=kernel,
                      shrinking=shrinking,
                      probability=probability)

    print('\n> SVC:')
    run_grid_search(model, X_train, y_train, param_grid, cv=10)


def fine_tune_xgb(X_train, y_train):

    model = xgb.XGBClassifier(subsample = 0.8, objective = 'binary:logistic', n_jobs = -1)

    learning_rate = [0.01, 0.1, 1]
    n_estimators = [100, 200, 300, 400]
    max_depth = [5, 10, 15]
    min_child_weight = range(0, 4)
    gamma = [0] #np.linspace(0, 0.9, 5)
    colsample_bytree = [0.6, 0.8, 1]
    scale_pos_weight = [1] #[0.1, 1]

    if small_search_space:
        learning_rate = [0.1]
        n_estimators = [200, 500]
        max_depth = [10]
        min_child_weight = [1]
        gamma = np.linspace(0, 0.1, 2)
        colsample_bytree = [1]
        scale_pos_weight = [1]

    param_grid = dict(learning_rate=learning_rate,
                      n_estimators=n_estimators,
                      max_depth=max_depth,
                      min_child_weight=min_child_weight,
                      gamma=gamma,
                      colsample_bytree=colsample_bytree,
                      scale_pos_weight=scale_pos_weight)

    print('\n> XGBoost:')
    run_grid_search(model, X_train, y_train, param_grid, cv=10)


if __name__ == '__main__':

    config_file = sys.argv[1]
    small_search_space = bool(int(sys.argv[2]))
    # config_file = Path('../../../config.yaml')
    # small_search_space = False

    cfg = Config(config_file)


    set_generator = PrepareTrainTestSets(cfg)


    data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')
    train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(data)

    # select random balanced dataset
    i = random.randint(0, len(train_dfs)-1)
    print(f"i: {i}")
    train_data = train_dfs[i]
    test_data = test_dfs[i]

    X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data,
                                                                                                    test_data)


    fine_tune_tree_based_models(X_train, y_train, clf_id='ExtraTreesClassifier')
    fine_tune_tree_based_models(X_train, y_train, clf_id='RandomForestClassifier')
    fine_tune_tree_based_models(X_train, y_train, clf_id='GradientBoostingClassifier')

    fine_tune_svc(X_train, y_train)

    fine_tune_xgb(X_train, y_train)

