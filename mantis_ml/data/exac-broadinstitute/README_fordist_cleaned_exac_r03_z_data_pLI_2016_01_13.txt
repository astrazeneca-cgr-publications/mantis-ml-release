README for fordist_cleaned_exac_r03_march16_z_pli_rec_null_data.txt
    Written by Kaitlin Samocha (samocha@broadinstitute.org)
    Last updated 2016 January 13.


These data were generated using: ExAC.r0.3.sites.vep.vcf.gz


The file (fordist_cleaned_exac_r03_march16_z_pli_rec_null_data.txt) contains the following information for the canonical transcript of genes as defined in Ensembl v75 (GENCODE v19):
- the probabilities of mutation
- observed and expected rare (minor allele frequency ~< 0.1%) single nucleotide variant counts
- Z scores for the deviation of observation from expectation
- the probability of being loss-of-function intolerant (intolerant of heterozygous and homozygous loss-of-function variants)
- the probability of being intolerant of homozygous, but not heterozygous loss-of-function variants
- the probability of being tolerant of both heterozygous and homozygous loss-of-function variants

Below, I have listed all column names with a brief description. Note that the observed variant counts require a VQSLOD >= -2.632, are not restricted to PASSing variants, and only reflect the number of unique single nucleotide variants as opposed to the allele count of said variants. The transcripts included have passed filtering criteria.


Creation of the Z score:
Briefly, we used a previously described — but slightly modified — sequence-context based mutational model to predict the number of expected rare (minor allele frequency < 0.1%) variants per transcript. We then calculated the chi-squared value for the deviation of observation from expectation for each mutational class (synonymous, missense, and loss-of-function). The square root of these values was taken and multiplied by -1 if the number of observed variants was greater than expectation (or 1 if observed counts were smaller than expected). The synonymous Z scores were then corrected by dividing each score by the standard deviation of all synonymous Z scores in between -5 and 5. For missense and loss-of-function Z scores, we took all Z scores between -5 and 0 and created a mirrored distribution. The missense and loss-of-function Z scores were then corrected by dividing each score by the standard deviation of these mirror distributions.

Higher Z scores indicate that the transcript is more intolerant of variation (more constrained).

For more information, see Samocha et al Nature Genetics 2014 (http://www.ncbi.nlm.nih.gov/pubmed/25086666).

*** WARNING ***
The loss-of-function Z score is highly correlated with gene length (r = 0.57). This means that longer genes have higher (more intolerant) scores simply because we are more confident in their predictions. To avoid this correlation, we include the probability of being loss-of-function intolerant (pLI), the suggested metric for evaluating a gene's intolerance (constraint) of loss-of-function variation.

Methods for the creation of pLI will be included in the supplement of the ExAC paper (Lek et al, in prep). Briefly, we used the observed and expected number of loss-of-function variants per gene to determine a posterior probability of each gene belonging to one of three categories:
1) completely tolerant of loss-of-function variation (observed = expected)
2) intolerant of two loss-of-function variants (like recessive genes, observed ~ 0.5*expected)
3) intolerant of a single loss-of-function variant (like haploinsufficient genes, observed ~ 0.1*expected)

pLI is the probability of falling into category 3, pRec is the probability of falling into category 2, and pNull is the probability of falling into category 1.


Filtering criteria:
- Dropped exon counts where the median_depth was < 1
- Dropped transcripts where no variants were observed at all (syn, mis, lof == 0)
- Dropped transcripts where syn Z > 3.71 and mis Z > 3.09 (considered outliers with too few variants)
- Dropped transcripts where syn Z < -3.71 and mis Z < -3.09 (considered outliers with too many variants)

These filters left 18,225 transcripts out of the original 19,621. Information for all transcripts will be released in the future.

Notes:
- If a canonical transcript is missing, it was likely covered too poorly to be evaluated.
- While I do not expect any more major changes, please understand that this is a work in progress.
- Loss-of-function (lof) here always refers to nonsense and essential splice site variants.
- If a transcript has 0 expected variants (and NAs for Z scores) that is because the depth of coverage for the transcript was very low. These transcripts are usually removed.


Columns:
transcript - Ensembl transcript ID
gene - Ensembl gene symbol
chr - chromosome
n_exons - number of exons in the transcript
cds_start - beginning of the transcript’s coding sequence if + strand, end if - strand
cds_end - end of the transcript’s coding sequence if + strand, beginning if - strand
bp - number of coding base pairs
mu_syn - probability of a synonymous mutation across the transcript
mu_mis - probability of a missense mutation across the transcript
mu_lof - probability of a loss-of-function mutation across the transcript
n_syn - number of rare (MAF < 0.1%) synonymous variants found in ExAC r0.3
n_mis - number of rare (MAF < 0.1%) missense variants found in ExAC r0.3
n_lof - number of rare (MAF < 0.1%) loss-of-function variants found in ExAC r0.3
exp_syn - depth adjusted number of expected rare (MAF < 0.1%) synonymous variants
exp_mis - depth adjusted number of expected rare (MAF < 0.1%) missense variants
exp_lof - depth adjusted number of expected rare (MAF < 0.1%) loss-of-function variants
syn_z - corrected synonymous Z score
mis_z - corrected missense Z score
lof_z - corrected loss-of-function Z score
pLI - the probability of being loss-of-function intolerant (intolerant of both heterozygous and homozygous lof variants)
pRec - the probability of being intolerant of homozygous, but not heterozygous lof variants
pNull - the probability of being tolerant of both heterozygous and homozygous lof variants
