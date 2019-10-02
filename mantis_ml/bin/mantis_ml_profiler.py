# -*- coding: utf-8 -*-
from collections import Counter
import argparse
import sys, os
import re

from mantis_ml.modules.pre_processing.data_compilation.process_features_filtered_by_disease import ProcessFeaturesFilteredByDisease
from mantis_ml.config_class import Config




# Disable print to stdout
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print to stdout
def enablePrint():
    sys.stdout = sys.__stdout__

def bordered(text):

    lines = text.splitlines()
    width = max(len(s) for s in lines)

    res = ['┌' + '─' * width + '┐']
    for s in lines:
        res.append('│' + (s + ' ' * width)[:width] + '│')
        res.append('└' + '─' * width + '┘')     
    
    return '\n'.join(res)


def assess_hpo_filtered_output(proc_obj, cfg, verbose=False):
    print(line_spacer + "-----------------   Assessing HPO filtering [config parameters: 'seed_include_terms']    -----------------\n")

    print("- Provided 'seed_include_terms':")
    print(cfg.seed_include_terms)

    seed_df, hpo_selected_terms = proc_obj.process_hpo(cfg.seed_include_terms, cfg.exclude_terms, cfg.phenotype)

    selected_genes = seed_df['Gene_Name'].tolist()
    if verbose:
        print('\n' + bordered(custom_bullet + ' Selected HPO genes ' + custom_bullet))
        print(selected_genes)

    print('\n' + bordered(custom_bullet + ' Selected HPO disease-associated terms ' + custom_bullet))
    print(sorted(list(hpo_selected_terms)))

    hpo_selected_terms_expanded = [s.split() for s in hpo_selected_terms]
    hpo_selected_terms_expanded = [item.lower() for sublist in hpo_selected_terms_expanded for item in sublist]

    # remove stopwords
    hpo_selected_terms_expanded = [w for w in hpo_selected_terms_expanded if w not in eng_stopwords]

    # remove digits
    hpo_selected_terms_expanded = [w for w in hpo_selected_terms_expanded if not w.isdigit()]

    if verbose:
        count_hpo_terms = Counter(hpo_selected_terms_expanded)
        print('\n' + u'\u2022' + ' Most common strings in filtered HPO phenotype terms:')
        for s, count in count_hpo_terms.most_common():
            print(s + ':', count)



def assess_gtex_filtered_output(proc_obj, cfg, verbose=False):
    print(line_spacer + "-----------------    Assessing GTEx filtering [config parameters: 'tissue' and 'additional_tissues']    -----------------\n")
    print("- Provided 'tissue':")
    print(cfg.tissue)
    print("\n- Provided 'additional_tissues':")
    print(cfg.additional_tissues)

    blockPrint()
    _, selected_tissue_cols, all_tissue_cols = proc_obj.process_gtex_features()
    enablePrint()

    all_tissue_cols = list(all_tissue_cols)
    selected_tissue_cols = list(selected_tissue_cols)
    
    if verbose:
        print('\nAvailable GTEx tissues:')
        print(sorted(all_tissue_cols))

    print('\n' + bordered(custom_bullet + ' Selected GTEx tissues ' + custom_bullet))
    print(sorted(selected_tissue_cols))



def assess_proteinatlas_filtered_output(proc_obj, cfg, verbose=False):
    print(line_spacer + "-----------------    Assessing Protein Atlas filtering [config parameters: 'tissue', 'seed_include_terms', 'additional_include_terms']    -----------------\n")
    print("- Provided 'tissue':")
    print(cfg.tissue)
    print("\n- Provided 'seed_include_terms':")
    print(cfg.seed_include_terms)
    print("\n- Provided 'additional_include_terms':")
    print(cfg.additional_include_terms)

    protatlas_include_terms = [cfg.tissue]
    protatlas_include_terms.extend(cfg.seed_include_terms)
    protatlas_include_terms.extend(cfg.additional_include_terms)

    blockPrint()
    prot_atlas_df, selected_normal_tissues, all_normal_tissues,  selected_rna_samples, all_rna_samples = proc_obj.process_protein_atlas_features(protatlas_include_terms, cfg.exclude_terms)
    enablePrint()

    if verbose:
        print('\nAvailable tissues (normal_tissue.tsv data):')
        print(sorted(all_normal_tissues))

    print('\n' + bordered(custom_bullet + ' Selected tissues from Protein Atlas (normal_tissue.tsv) ' + custom_bullet))
    print(sorted(selected_normal_tissues))

    if verbose:
        print('\nAvailable samples (rna_tissue.tsv data):')
        print(sorted(all_rna_samples))

    print('\n' + bordered(custom_bullet + ' Selected samples from Protein Atlas (rna_tissue.tsv) ' + custom_bullet))
    print(sorted(selected_rna_samples))




def assess_msigdb_filtered_output(proc_obj, cfg, verbose=False):
    print(line_spacer + "-----------------    Assessing MSigDB filtering [config parameters: 'tissue', 'seed_include_terms', 'additional_include_terms']    -----------------\n")
    print("- Provided 'tissue':")
    print(cfg.tissue)
    print("\n- Provided 'seed_include_terms':")
    print(cfg.seed_include_terms)
    print("\n- Provided 'additional_include_terms':")
    print(cfg.additional_include_terms)

    msigdb_include_terms = [cfg.tissue]
    msigdb_include_terms.extend(cfg.seed_include_terms)
    msigdb_include_terms.extend(cfg.additional_include_terms)
    exclude_terms = cfg.exclude_terms

    if cfg.generic_classifier:
        msigdb_include_terms = ['.*']
        exclude_terms = []

    blockPrint()
    msigdb_go_df, selected_go_terms = proc_obj.process_msigdb_go_features(msigdb_include_terms, exclude_terms)
    enablePrint()

    print('\n' + bordered(custom_bullet + ' Selected Gene Ontology terms (from MSigDB) ' + custom_bullet))
    print(sorted(selected_go_terms))



def assess_mgi_filtered_output(proc_obj, cfg, verbose=False):
    if cfg.generic_classifier:
        return 0

    print(line_spacer + "-----------------    Assessing MGI filtering [config parameters: 'seed_include_terms', 'additional_include_terms']    -----------------\n")
    print("- Provided 'seed_include_terms':")
    print(cfg.seed_include_terms)
    print("\n- Provided 'additional_include_terms':")
    print(cfg.additional_include_terms)

    mgi_include_terms = cfg.seed_include_terms
    mgi_include_terms.extend(cfg.additional_include_terms)

    blockPrint()
    _, selected_mgi_phenotypes = proc_obj.process_mgi_features(mgi_include_terms, cfg.exclude_terms)
    enablePrint()
    # print(selected_mgi_phenotypes)

    selected_mgi_phenotypes_expanded = [s.split('|') for s in selected_mgi_phenotypes]
    selected_mgi_phenotypes_expanded = list(set([item.lower() for sublist in selected_mgi_phenotypes_expanded for item in sublist]))
    # print(selected_mgi_phenotypes_expanded)

    include_pattern = re.compile('|'.join(mgi_include_terms), re.IGNORECASE)

    filtered_selected_mgi_phenotypes_expanded = list(filter(lambda x: re.search(include_pattern, x), selected_mgi_phenotypes_expanded))
    # print(filtered_selected_mgi_phenotypes_expanded)

    print('\n' + bordered(custom_bullet + ' Selected MGI phenotypes ' + custom_bullet))
    print(sorted(filtered_selected_mgi_phenotypes_expanded))



def get_english_stopwords():
    eng_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    return eng_stopwords


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('config_file') 
    parser.add_argument('-v', '--verbosity', action="count", help="print verbose output verbosity (run with -v option)")      
    args = parser.parse_args()

    if args.verbosity:
        verbose = True
    else:
        verbose = False
    print('>>> Running mantis-ml config profiling ...')
    print('verbose:', verbose)
    print('Config file:', args.config_file)

    cfg = Config(args.config_file)

    proc_obj = ProcessFeaturesFilteredByDisease(cfg)

    # common strings to exclude from profiling
    eng_stopwords = get_english_stopwords()
    custom_bullet = u'\u2022' * 5
    line_spacer = '\n' * 6

    # HPO
    assess_hpo_filtered_output(proc_obj, cfg, verbose=verbose)


    # GTEx
    assess_gtex_filtered_output(proc_obj, cfg, verbose=verbose)


    # Protein Atlas
    assess_proteinatlas_filtered_output(proc_obj, cfg, verbose=verbose)


    # MSigDB
    assess_msigdb_filtered_output(proc_obj, cfg, verbose=verbose)

    # MGI
    assess_mgi_filtered_output(proc_obj, cfg, verbose=verbose)

    print('\n\n<<< mantis-ml config profiling complete.')
