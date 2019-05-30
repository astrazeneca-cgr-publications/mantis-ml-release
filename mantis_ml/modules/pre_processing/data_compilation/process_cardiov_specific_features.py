import pandas as pd
import sys

from mantis_ml.modules.pre_processing.data_compilation.process_generic_features import ProcessGenericFeatures
from mantis_ml.config_class import Config


class ProcessCardiovascularSpecificFeatures(ProcessGenericFeatures):

    def __init__(self, cfg):
        ProcessGenericFeatures.__init__(self, cfg)

    def get_exsnp_features(self):
        print("\n>> Compiling exSNP features")

        cad_df = pd.read_csv(self.cfg.data_dir / 'exSNP/CAD_eQTL_hits.csv')
        ht_df = pd.read_csv(self.cfg.data_dir / 'exSNP/HT_eQTL_hits.csv')
        print(cad_df.shape)
        print(ht_df.shape)
        exsnp_df = pd.merge(cad_df, ht_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
        exsnp_df.fillna(0, inplace=True)

        return exsnp_df

    def get_platelet_eqtl_feature(self):
        print("\n>> Compiling Platelet eQTL features")

        platelet_eqtl_df = pd.read_csv(self.cfg.data_dir / 'platelets_eqtl/platelets_eQTL.csv')

        return platelet_eqtl_df


    def get_adipose_eqtl_features(self):
        print("\n>>Compiling Adipose eQTL features...")

        adipose_eqtl_df = pd.read_csv(self.cfg.data_dir / 'adipose_eqtl/adipose_eQTL.csv')

        return adipose_eqtl_df


    def run_all(self):
        exsnp_df = self.get_exsnp_features()
        print('ExSNP:', exsnp_df.shape)

        platelet_eqtl_df = self.get_platelet_eqtl_feature()
        print('Platelets eQTL:', platelet_eqtl_df.shape)

        adipose_eqtl_df = self.get_adipose_eqtl_features()
        print('Adipose eQTL:', adipose_eqtl_df.shape)


        print("\n>> Merging all data frames together...")
        cardiov_specific_features_df = pd.merge(exsnp_df, platelet_eqtl_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
        print(cardiov_specific_features_df.shape)
        cardiov_specific_features_df = pd.merge(cardiov_specific_features_df, adipose_eqtl_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
        print(cardiov_specific_features_df.shape)



        default_gene_set = self.cfg.hgnc_genes_series
        tmp_df = pd.DataFrame({'Gene_Name': default_gene_set})
        cardiov_specific_features_df = pd.merge(cardiov_specific_features_df, tmp_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
        # Impute 'cardiov_specific_features_df' features with zero, for all genes that don't have a '1' value:
        # these values are not missing data but rather represent a 'False'/zero feature value.
        cardiov_specific_features_df.fillna(0, inplace=True)


        cardiov_specific_features_df.to_csv(self.cfg.cardiov_specific_feature_table, sep='\t', index=None)
        print("Saved to {0}".format(self.cfg.cardiov_specific_feature_table))


        print('Duplicates:', cardiov_specific_features_df[ cardiov_specific_features_df.Gene_Name.duplicated() ])

if __name__ == '__main__':

    config_file = '../../../config.yaml'
    cfg = Config(config_file)

    proc = ProcessCardiovascularSpecificFeatures(cfg)
    proc.run_all()