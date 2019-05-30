import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 100)
import re
import sys
from homogenise_features_from_different_diseases import homogenise_tissue_specific_features


ckd_boruta_df = pd.read_csv('../out/CKD-production/feature_selection/boruta/merged.boruta_imp_df.txt', header=0)
ckd_boruta_df = pd.DataFrame(ckd_boruta_df.mean(axis=0), columns=['CKD']).sort_values(by='CKD', ascending=False)
#print(ckd_boruta_df.index.values)
ckd_boruta_df = homogenise_tissue_specific_features(ckd_boruta_df)

epilepsy_boruta_df = pd.read_csv('../out/Epilepsy-production/feature_selection/boruta/merged.boruta_imp_df.txt', header=0)
epilepsy_boruta_df = pd.DataFrame(epilepsy_boruta_df.mean(axis=0), columns=['Epilepsy']).sort_values(by='Epilepsy', ascending=False)
#print(epilepsy_boruta_df.index.values)
epilepsy_boruta_df = homogenise_tissue_specific_features(epilepsy_boruta_df)

als_boruta_df = pd.read_csv('../out/ALS-production/feature_selection/boruta/merged.boruta_imp_df.txt', header=0)
als_boruta_df = pd.DataFrame(als_boruta_df.mean(axis=0), columns=['ALS']).sort_values(by='ALS', ascending=False)
#print(als_boruta_df.index.values)
als_boruta_df = homogenise_tissue_specific_features(als_boruta_df)



generic_boruta_df = pd.read_csv('../out/Generic-production/feature_selection/boruta/merged.boruta_imp_df.txt', header=0)
generic_boruta_df = pd.DataFrame(generic_boruta_df.mean(axis=0), columns=['Generic']).sort_values(by='Generic', ascending=False)
generic_boruta_df = homogenise_tissue_specific_features(generic_boruta_df)

generic_boruta_df.reset_index(inplace=True)
generic_boruta_df.columns.values[0] = 'Features'

sorted_generic_boruta_df = generic_boruta_df.sort_values(by='Generic', ascending=False)
print(sorted_generic_boruta_df.head(20))
sys.exit()


disease_df = ckd_boruta_df.merge(epilepsy_boruta_df, left_index=True, right_index=True, how='inner')
disease_df = disease_df.merge(als_boruta_df, left_index=True, right_index=True, how='inner')
print(disease_df.head())
print(disease_df.shape)


colors_dict = {'CKD': '#377eb8', 'Epilepsy': '#4daf4a', 'ALS': '#984ea3', 'Disease sum': 'orange', 'Generic': 'black'}

def make_grouped_barplot(df, disease_cols, out_filename):

    print(disease_cols)

    # Setting the positions and width for the bars 
    pos = list(range(len(df[disease_cols[0]]))) 
    width = 0.25 

    # Plotting the bars 
    fig, ax = plt.subplots(figsize=(15,5))

    # Create a bar with CKD data, in position pos:
    alpha_param = 0.9
    for i in range(len(disease_cols)):
        cur_col = disease_cols[i]
        plt.bar([p + width*i for p in pos], df[cur_col], width, alpha=alpha_param, color=colors_dict[cur_col], label=cur_col)

    ax.set_ylabel('Normalised Average Boruta scores')
    ax.set_title('Boruta Feautre Importance scores across different Diseases')
    ax.set_xticks([p + 1.5 * width for p in pos])
    ax.set_xticklabels(df['Features'], rotation=90, fontsize=10)

    plt.legend(loc='upper right', fontsize=15)

    fig.savefig(out_filename + '.pdf', bbox_inches='tight')


# mean normalisation:
#disease_df = (disease_df - disease_df.mean()) / disease_df.std()
# min-max normalisation:
disease_df = (disease_df - disease_df.min()) / (disease_df.max() - disease_df.min())
disease_df.reset_index(inplace=True)
disease_df.columns.values[0] = 'Features'

# filter out shadow features
valid_features = [f for f in disease_df['Features'] if 'shadow' not in f]
disease_df = disease_df.loc[ disease_df['Features'].isin(valid_features), :]

disease_df['Disease sum'] = disease_df['CKD'] + disease_df['Epilepsy'] + disease_df['ALS']
disease_df.sort_values(by='Disease sum', inplace=True, ascending=False)
top_features = 20
top_disease_df = disease_df.iloc[:top_features, :]

    
disease_cols = [c for c in disease_df.columns if c not in ['Features', 'Disease sum']]
make_grouped_barplot(top_disease_df, disease_cols, 'Disease_boruta_grouped_barplots')


# Generic vs Avg Disease boruta feature importance scores
print(disease_df['Features'])
print(generic_boruta_df['Features'])
gen_disease_df = disease_df.merge(generic_boruta_df, left_on='Features', right_on='Features', how='inner')
print(gen_disease_df['Features'])
print(gen_disease_df.head())

gen_disease_df = gen_disease_df[['Features', 'Disease sum', 'Generic']]
gen_disease_feature_col = gen_disease_df['Features']
gen_disease_df.drop(['Features'], axis=1, inplace=True)
gen_disease_df = (gen_disease_df - gen_disease_df.min()) / (gen_disease_df.max() - gen_disease_df.min()) 
gen_disease_df['Features'] = gen_disease_feature_col
print(gen_disease_df.head())

gen_disease_df = gen_disease_df.iloc[:top_features, :]

disease_cols = ['Disease sum', 'Generic']
make_grouped_barplot(gen_disease_df, disease_cols, 'Generic_vs_Avg_Disease_boruta_grouped_barplots')
