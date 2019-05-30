import numpy as np
import pandas as pd
from mantis_ml.config_class import Config


class HGNCtoEnsemblIDMapper:

    def __init__(self, cfg):
        self.cfg = cfg


    def run(self):
        hgnc_to_ens_gids_df = pd.read_csv('../../../../data/ensembl/hgnc_to_ens_gids.txt', sep='\t', header=None)
        hgnc_to_ens_gids_df.columns = ['HGNC', 'ENS_GID']
        print(hgnc_to_ens_gids_df.head())
        print(hgnc_to_ens_gids_df.shape)

        hgnc_to_ens_gids_df = hgnc_to_ens_gids_df.loc[ hgnc_to_ens_gids_df.HGNC.isin(self.cfg.hgnc_genes_series)]
        print(hgnc_to_ens_gids_df.shape)


        # Manually add ENS gene ids for unmapped HGNC entries: from Ensembl and genecards
        # TODO: use file with all HGNC synonyms to then map against Ensembl Gene Ids
        unmapped_hgnc_gids = set(self.cfg.hgnc_genes_series) - set(hgnc_to_ens_gids_df['HGNC'])
        print(unmapped_hgnc_gids)

        suppl_dict = {'NIM1': 'ENSG00000177453', 'PHF17': 'ENSG00000077684', 'PHF16': 'ENSG00000102221', 'NEURL': 'ENSG00000107954',
                    'RP11-389E17.1': 'ENSG00000231171', 'JHDM1D': 'ENSG00000006459', 'CXorf48': 'ENSG00000169551',
                    'SELRC1': 'ENSG00000162377', 'CTC-349C3.1': 'ENSG00000249647', 'C10orf137': 'ENSG00000107938',
                    'PHF15': 'ENSG00000043143', 'AC007390.5': 'ENSG00000218739', 'RP11-332O19.5': 'ENSG00000237827',
                    'CTC-360G5.1': 'ENSG00000262484', 'PLAC1L': 'ENSG00000149507', 'KIAA1737': 'ENSG00000198894',
                    'KIAA1984': 'ENSG00000213213', 'C7orf41': 'ENSG00000180354', 'C12orf52': 'ENSG00000139405',
                    'MLF1IP': 'ENSG00000151725', 'RP11-57H12.6': 'ENSG00000271092', 'MKI67IP': 'ENSG00000155438',
                    'C7orf10': 'ENSG00000175600', 'CXorf61': 'ENSG00000204019'}


        suppl_df = pd.DataFrame(pd.Series(suppl_dict)).reset_index()
        suppl_df.columns = ['HGNC', 'ENS_GID']
        print(suppl_df.head())

        hgnc_to_ens_gids_df = pd.concat([hgnc_to_ens_gids_df, suppl_df], axis=0)
        print(hgnc_to_ens_gids_df.shape)

        hgnc_to_ens_gids_df['ENS_GID'].to_csv('../../../../data/ensembl/ens_gids_for_promoter_check.txt', index=None)

if __name__ == '__main__':

    config_file = '../../../config.yaml'
    cfg = Config(config_file)

    id_mapper = HGNCtoEnsemblIDMapper(cfg)
    id_mapper.run()