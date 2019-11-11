import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from mantis_ml.config_class import Config


best_real_clf = {'CKD': 'XGBoost', 'Epilepsy': 'XGBoost', 'ALS': 'ExtraTreesClassifier'}
best_rand_clf = {'CKD': 'XGBoost', 'Epilepsy': 'XGBoost', 'ALS': 'Stacking'}


def run(disease):
    
    base_dir = '../../../data/random_vs_seed-mantis_ml_scores/' + disease
    
    real_known_df = pd.read_csv(base_dir + '/real_seeds.' + best_real_clf[disease] + '.Known_genes.Ranked_by_prediction_proba.csv', header=None)
    random_known_df = pd.read_csv(base_dir + '/random_seeds.' + best_rand_clf[disease] + '.Known_genes.Ranked_by_prediction_proba.csv', header=None)

    real_seed_genes = real_known_df.iloc[:, 0].tolist()
    random_seed_genes = random_known_df.iloc[:, 0].tolist()

#     print(real_seed_genes[:10])
#     print(random_seed_genes[:10])


    real_df = pd.read_csv(base_dir + '/real_seeds.' + best_real_clf[disease] + '.All_genes.mantis-ml_percentiles.csv', index_col=0)
    random_df = pd.read_csv(base_dir + '/random_seeds.' + best_rand_clf[disease] + '.All_genes.mantis-ml_percentiles.csv', index_col=0)
    real_df.head()

    known_real_perc = real_df.loc[real_df.Gene_Name.isin(real_seed_genes), :]
    known_random_perc = random_df.loc[random_df.Gene_Name.isin(random_seed_genes), :]

#     print(real_df.head())
#     print(known_random_perc.head())
#     print(known_random_perc.shape)
    
    fig, ax = plt.subplots(figsize=(12,8))
    sns.distplot(known_real_perc['mantis_ml_perc'].tolist(), hist=False, kde=True, 
                 color = 'darkblue',
                 kde_kws={'linewidth': 2}, label='real seed genes')

    sns.distplot(known_random_perc['mantis_ml_perc'].tolist(), hist=False, kde=True, 
                 color = 'orange',
                 kde_kws={'linewidth': 2}, label='random seed genes')


    res =  mannwhitneyu(known_real_perc['mantis_ml_perc'], known_random_perc['mantis_ml_perc'])
    print(res)


    ax.legend(fontsize=12)
    plt.text(-0.3, 0.03, 'Mann-Whitney U: ' + str(res.pvalue), fontsize=11)
    plt.title('CKD')

    fig.savefig(base_dir + '/real_vs_random_seeds-density_plots.pdf')
    
    
run('CKD')

run('Epilepsy')

run('ALS')
