import pandas as pd
import re
import sys


def homogenise_tissue_specific_features(df, reset_index=True):

    if reset_index:
        df.reset_index(inplace=True)
        df.columns.values[0] = 'Features'

    # ===== Edit "GTEx.*expression" features =====
    gtex_expression_features = [c for c in df['Features'] if re.search(re.compile('GTEx.*expression'), c)]
    for f in gtex_expression_features:
        df.loc[ df['Features'] == f, 'Features'] = 'GTEx_tissue_specific_TPM_expression'

    # ===== Edit "GTEx.*Rank" features =====
    gtex_rank_features = [c for c in df['Features'] if re.search(re.compile('GTEx.*Rank'), c)]
    for f in gtex_rank_features:
        df.loc[ df['Features'] == f, 'Features'] = 'GTEx_tissue_specific_Expression_Rank'
   
    # ===== Edit "GO_.*" features =====
    go_features = [c for c in df['Features'] if re.search(re.compile('GO_.*'), c)]
    for f in go_features:
        df.loc[ df['Features'] == f, 'Features'] = 'GO_tissue_specific'

    # ===== Edit ".*_GWAS_.*" features =====
    tissue_gwas_features = [c for c in df['Features'] if re.search(re.compile('.*_GWAS_.*'), c)]
    gwas_tissues = list(set([f.split('_')[0] for f in tissue_gwas_features]))
    for t in gwas_tissues:
        df['Features'] = df['Features'].str.replace(t + '_GWAS', 'Tissue_specific_GWAS')

    # >> Relevant for generic classifier results only
    # ===== Edit "ProteinAtlas_.*Expr_Flag" features =====
    proteinatlas_expr_flags = [c for c in df['Features'] if re.search(re.compile('ProteinAtlas_.*Expr_Flag'), c)]
    for f in proteinatlas_expr_flags:
        df.loc[ df['Features'] == f, 'Features'] = 'ProteinAtlas_gene_expr_levels_Not_detected'

    # ===== Edit "ProteinAtlas_.*_RNA_Expr_TPM" features =====
    proteinatlas_expr_tpms = [c for c in df['Features'] if re.search(re.compile('ProteinAtlas_.*_RNA_Expr_TPM'), c)]
    for f in proteinatlas_expr_tpms:
        df.loc[ df['Features'] == f, 'Features'] = 'ProteinAtlas_RNA_expression_TPM'
   
    
    # correct typo: TMP -> TPM
    df['Features'] = df['Features'].str.replace('ProteinAtlas_RNA_expression_TMP', 'ProteinAtlas_RNA_expression_TPM')

    df = df.groupby(['Features'])[df.columns.values].max()

    if reset_index:
        df.drop(['Features'], axis=1, inplace=True)
    

    return df

