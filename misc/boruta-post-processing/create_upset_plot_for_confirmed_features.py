import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys 
import pyupset as pyu
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
from homogenise_features_from_different_diseases import homogenise_tissue_specific_features


ckd_confirmed_df = pd.read_csv('../out/CKD-production/feature_selection/boruta/Confirmed.boruta_features.csv', header=None)
ckd_confirmed_df.columns = ['Features']
ckd_confirmed_df = homogenise_tissue_specific_features(ckd_confirmed_df, reset_index=False)
#print(ckd_confirmed_df)

epilepsy_confirmed_df = pd.read_csv('../out/Epilepsy-production/feature_selection/boruta/Confirmed.boruta_features.csv', header=None)
epilepsy_confirmed_df.columns = ['Features']
epilepsy_confirmed_df = homogenise_tissue_specific_features(epilepsy_confirmed_df, reset_index=False)
#print(epilepsy_confirmed_df.head())

als_confirmed_df = pd.read_csv('../out/ALS-production/feature_selection/boruta/Confirmed.boruta_features.csv', header=None)
als_confirmed_df.columns = ['Features']
als_confirmed_df = homogenise_tissue_specific_features(als_confirmed_df, reset_index=False)
#print(als_confirmed_df.head())

generic_confirmed_df = pd.read_csv('../out/Generic-production/feature_selection/boruta/Confirmed.boruta_features.csv', header=None)
generic_confirmed_df.columns = ['Features']
generic_confirmed_df = homogenise_tissue_specific_features(generic_confirmed_df, reset_index=False)
#print(generic_confirmed_df.head())


confirmed_features_dict = {}
confirmed_features_dict['CKD'] = ckd_confirmed_df
confirmed_features_dict['Epilepsy'] = epilepsy_confirmed_df
confirmed_features_dict['ALS'] = als_confirmed_df
confirmed_features_dict['Generic'] = generic_confirmed_df


min_inters_size = 1
pyu.plot(confirmed_features_dict, sort_by='degree', inters_size_bounds=(min_inters_size, np.inf))

cur_fig = matplotlib.pyplot.gcf()
cur_fig.savefig('Confirmed_features_intersection_between_classifiers.pdf', bbox_inches='tight')


# === Print intersection / union sets ===
# Degree 4
intersection_disease_features = list(set(ckd_confirmed_df['Features'].tolist()) & set(epilepsy_confirmed_df['Features'].tolist()) & set(als_confirmed_df['Features'].tolist()))
#print('intersection_disease_features:', intersection_disease_features)
disease_generic_intersection = list(set(intersection_disease_features) & set(generic_confirmed_df['Features'].tolist()) )

print('\n>Degree:4 -- disease_generic_intersection:', disease_generic_intersection)
seen_features = set(disease_generic_intersection)


# Degree 3
generic_ckd_epilepsy_features = list(set(ckd_confirmed_df['Features'].tolist()) & set(epilepsy_confirmed_df['Features'].tolist()) & set(generic_confirmed_df['Features'].tolist()))
generic_ckd_epilepsy_features = list(set(generic_ckd_epilepsy_features) - seen_features)
print('\n>Degree: 3 -- generic_ckd_epilepsy_features:', generic_ckd_epilepsy_features)
seen_features = seen_features | set(generic_ckd_epilepsy_features)

intersection_disease_only_features = list(set(intersection_disease_features) - seen_features)
print('>Degree: 3 -- intersection_disease_only_features:', intersection_disease_only_features)
seen_features = seen_features | set(intersection_disease_only_features)


# Degree 2
generic_epilepsy_features = list(set(epilepsy_confirmed_df['Features'].tolist()) & set(generic_confirmed_df['Features'].tolist()) - seen_features)
print('\n>Degree: 2 -- generic_epilepsy_features:', generic_epilepsy_features)
seen_features = seen_features | set(generic_epilepsy_features)

generic_ckd_features = list(set(ckd_confirmed_df['Features'].tolist()) & set(generic_confirmed_df['Features'].tolist()) - seen_features)
print('>Degree 2: -- generic_ckd_features:', generic_ckd_features)
seen_features = seen_features | set(generic_ckd_features)

ckd_epilepsy_features = list(set(ckd_confirmed_df['Features'].tolist()) & set(epilepsy_confirmed_df['Features'].tolist()) - seen_features)
print('>Degree: 2 -- ckd_epilepsy_features:', ckd_epilepsy_features)
seen_features = seen_features | set(ckd_epilepsy_features)


# Degree 1
generic_only_features = list(set(generic_confirmed_df['Features'].tolist()) - seen_features)
print('\n>Degree: 1 -- generic_only_features:', generic_only_features)
seen_features = seen_features | set(generic_only_features)

epilepsy_only_features = list(set(epilepsy_confirmed_df['Features'].tolist()) - seen_features)
print('>Degree: 1 -- epilepsy_only_features:', epilepsy_only_features)
seen_features = seen_features | set(epilepsy_only_features)

ckd_only_features = list(set(ckd_confirmed_df['Features'].tolist()) - seen_features)
print('>Degree: 1 -- ckd_only_features:', ckd_only_features)
seen_features = seen_features | set(ckd_only_features)
