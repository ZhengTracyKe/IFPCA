# 8 Single-cell RNA-seq data sets
This folder contains 8 Single-cell RNA-seq data sets studied in the paper *D. Chen, J, Jin, and Z.T. Ke (2023) Subject clustering by IF-PCA and several recent methods*.

Data matrix (p-by-n) is the log-counts of the single-cell RNA-sequence reads of different genes (features) in different cells(samples). The data were downloaded and processed from the Hemberg Group at the Sanger Institute (https://hemberg-lab.github.io/scRNA.seq.datasets).

## Name set: (p-by-n data matrix)
### 1. camp1, 13111-by-777
### 2. camp2, 11233-by-734
### 3. darmanis, 13400-by-466
### 4. deng, 16347-by-268
### 5. goolam, 21199-by-124
### 6. grun, 5547-by-1502
### 7. li, 25369-by-561
### 8. patel, 5948-by-430

## Each data set has two files:

### *name-x-filter.txt*: p-by-n data matrix after the pre-processing (features (genes) with fractions of non- zero entries < 5% are filtered out).

### *name-y.txt*: n-by-1 true class labels (start from 0)

