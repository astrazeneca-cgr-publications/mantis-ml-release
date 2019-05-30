import pandas as pd
import sys

from mantis_ml.modules.pre_processing.data_compilation.process_generic_features import ProcessGenericFeatures
from mantis_ml.config_class import Config


class ProcessCKDSpecificFeatures(ProcessGenericFeatures):

    def __init__(self, cfg):
        ProcessGenericFeatures.__init__(self, cfg)

    def get_ckddb_features(self):
    
        print("\n>> Compiling CKDdb features")
    
        ckddb_df = pd.read_csv(self.cfg.data_dir / 'ckddb/CKDdb_annotation_features.tsv', sep='\t')
        ckddb_df.CKDdb_Disease = 1
    
        return ckddb_df
    
    def get_goa_kidney_research_priority_feature(self):
        print("\n>> Compiling GOA Kidney_Research_Priority feature...")
        goa_features_df = pd.read_csv(self.cfg.data_dir / 'goa/GOA_Kidney_Research_Priority_feature.tsv', sep='\t')
        goa_features_df.GOA_Kidney_Research_Priority.replace({'Yes': 1, 'No':0}, inplace=True)
    
        return goa_features_df
    
    def get_neph_qtl_features(self):
        print("\n>>Compiling nephro eQTL features...")
    
        print("- Tubuli QTL features")
        tub_qtl_df = pd.read_csv(self.cfg.data_dir / 'neph_qtl/tub_feature_table.csv')
        print(tub_qtl_df.shape)
    
        print("- Glomerular QTL features")
        glom_qtl_df = pd.read_csv(self.cfg.data_dir / 'neph_qtl/glom_feature_table.csv')
        print(glom_qtl_df.shape)
    
        neph_qtl_df = pd.merge(tub_qtl_df, glom_qtl_df, left_on='Gene_Name', right_on='Gene_Name', how='outer')
    
        return neph_qtl_df
    
    
    def run_all(self):
        ckddb_df = self.get_ckddb_features()
        print('CKDdb:', ckddb_df.shape)
    
        goa_features_df = self.get_goa_kidney_research_priority_feature()
        print('GOA:', goa_features_df.shape)
    
        neph_qtl_df = self.get_neph_qtl_features()
        print('NephQTL:', neph_qtl_df.shape)
    

        print("\n>> Merging all data frames together...")
        ckd_specific_features_df = pd.merge(ckddb_df, goa_features_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
        print(ckd_specific_features_df.shape)
        ckd_specific_features_df = pd.merge(ckd_specific_features_df, neph_qtl_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
        print(ckd_specific_features_df.shape)

        # Impute neph_qtl-features and 'GOA_Kidney_Research_Priority' with zero, for all genes that don't have a '1' value:
        # these values are not missing data but rather represent a 'False'/zero feature value.
        neph_qtl_cols = [c for c in neph_qtl_df.columns.values if c != 'Gene_Name']
        for c in neph_qtl_cols:
            ckd_specific_features_df[c].fillna(0, inplace=True)
        ckd_specific_features_df['GOA_Kidney_Research_Priority'].fillna(0, inplace=True)


        ckd_specific_features_df.to_csv(self.cfg.ckd_specific_feature_table, sep='\t', index=None)
        print("Saved to {0}".format(self.cfg.ckd_specific_feature_table))

        print(ckd_specific_features_df.isna().sum())
        print(ckd_specific_features_df.head())
        print(ckd_specific_features_df.shape)

if __name__ == '__main__':

    config_file = '../../../config.yaml'
    cfg = Config(config_file)

    proc = ProcessCKDSpecificFeatures(cfg)
    proc.run_all()