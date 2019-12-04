## mantis-ml

A high-level machine learning framework for gene prioritisation implemented on top of `scikit-learn` and `keras/tensorflow`.

<br>

### `mantisml`
```
usage: mantisml [-h] -c CONFIG_FILE -o OUTPUT_DIR
                [-r {all,pre,boruta,pu,post,post_unsup}] [-k KNOWN_GENES_FILE]
                [-n NTHREADS] [-i ITERATIONS] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE        Config file (.yaml) with run parameters [Required]

  -o OUTPUT_DIR         Output directory name
                        (absolute/relative path e.g. ./CKD, /tmp/Epilepsy-testing, etc.)
                        If it doesn't exist it will automatically be created [Required]

  -r {all,pre,boruta,pu,post,post_unsup}
                        Specify type of analysis to run (default: all)

  -k KNOWN_GENES_FILE   File with custom list of known genes used for training (new-line separated)

  -n NTHREADS           Number of threads (default: 4)

  -i ITERATIONS         Number of stochastic iterations for semi-supervised learning (default: 10)

  -s, --stacking        Include 'Stacking' in set of classifiers
```

<br>

### `mantisml-overlap`
```
usage: mantisml-overlap [-h] -c CONFIG_FILE -o OUTPUT_DIR -e
                        EXTERNAL_RANKED_FILE [-t TOP_RATIO]
                        [-m MAX_OVERLAPPING_GENES] [-y YLIM] [-f]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE        Config file (.yaml) with run parameters [Required]

  -o OUTPUT_DIR         Output directory name
                        (absolute/relative path e.g. ./CKD, /tmp/Epilepsy-testing, etc.)
                        If it doesn't exist it will automatically be created [Required]

  -e EXTERNAL_RANKED_FILE
                        Input file with external ranked gene list;
                        either 1-column or 2-columns (with p-values in the 2nd column) [Required]

  -t TOP_RATIO          Top percent ratio of mantis-ml predictions
                        to overlap with the external ranked list (default: 5)

  -m MAX_OVERLAPPING_GENES
                        Max. number of genes to retain that overlap
                        mantis-ml and EXTERNAL_RANKED_FILE predictions (default: 50)

  -y YLIM               Explicitly define y-axis max. limit (PHRED score value)

  -f, --full_xaxis      Plot enrichment signal across the entire x-axis
                        and not just for the significant part (or the MAX_OVERLAPPING_GENES)
                        of the external ranked list
```
  
<br>
  
### `mantisml-profiler`
```
usage: mantisml-profiler [-h] -c CONFIG_FILE -o OUTPUT_DIR [-v]

optional arguments:
  -h, --help       show this help message and exit
  -c CONFIG_FILE   Config file (.yaml) with run parameters [Required]

  -o OUTPUT_DIR    Output directory name
                   (absolute/relative path e.g. ./CKD, /tmp/Epilepsy-testing, etc.)
                   If it doesn't exist it will automatically be created [Required]

  -v, --verbosity  Print verbose output
```
