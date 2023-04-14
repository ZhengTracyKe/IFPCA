###### Run SC3 on the genetic data sets

#### This file is run on RStudio under R 3.6.3, need to install Bioconductor 3.10 to load 'SingleCellExperiment' and 'SC3' as follows:
##check corresponding Bioconductor version at: https://www.bioconductor.org/about/release-announcements/:
#if (!require("BiocManager", quietly = TRUE))
# install.packages("BiocManager")
#BiocManager::install(version = "3.10") #install bioconductor
#BiocManager::install('SingleCellExperiment', force = TRUE) 
## or install SC3 from Github:
#library('devtools')
#install_github("hemberg-lab/SC3")

## SC3 Code reference: https://nbisweden.github.io/workshop-archive/workshop-scRNAseq/2018-05-21/labs/sc3_ilc.html
#### the result is stable, one iteration is enough

library('SingleCellExperiment')
library('SC3') #in Bioconductor
library('aricode') # ARI

### Load Microarray data sets
## brain, K = 5, n = 42, p =5597
## breast, K = 2, n = 276, p =22,215
## colon, K = 2, n = 62, p =2000
## leukemia, K = 2, n = 72, p =3571
## lung1, K = 2, n = 181, p =12,533
## lung2, K = 2, n = 203, p = 12,600
## lymphoma, K = 3, n = 62, p = 4,029
## prostate, K = 2, n = 102, p = 6033
## srbct, K = 4, n = 63, p = 2308
## su, K = 2, n = 174, p = 7909

#X = read.table("breast.x.txt", sep=',',header = FALSE) #p-by-n 
#X = read.table("lung1.x.txt", sep=',',header = FALSE) #p-by-n 
#X = read.table("lung2.x.txt", sep=',',header = FALSE) #p-by-n 
#X = read.table("su.x.txt", sep=',',header = FALSE) #p-by-n 
## for all other data sets, use the following:
name = 'brain'
X = read.table(paste0(name, ".x.txt"), header = FALSE) #p-by-n
Y = as.numeric(read.table(paste0(name, ".y.txt"), header = FALSE)$V1) + 1 #start from 1, true cluster label


### Load Sinlge-Cell RAN-seq data
## load pre-processed(>5% non-zero entry for each feature) data
#camp1, K = 7,13111*777
#camp2, K = 6, 11233*734
#darmanis, K = 9, 13400*466
#deng, K = 6,16347*268
#goolam, K = 5, 21199*124
#grun,K = 2, 5547*1502
#li, K = 9, 25369*561
#patel, K = 5, 5948*430
name = 'patel'
X = read.table(paste0(name, "-x-filter.txt"), sep = ',', header = FALSE) #p-by-n
Y = as.numeric(read.table(paste0(name, "-y.txt"), header = FALSE)$V1) + 1 #start from 1, true cluster label



### Create SingleCellExperiment Object, input p-by-n matrix
sce_X <- SingleCellExperiment(assays = list(counts = exp(as.matrix(X))-1,logcounts = as.matrix(X)), rowData = DataFrame(feature_symbol = 1:dim(X)[1]))
#### Run SC3 
## this step may take 1-5 mins depending on the data size, can do gene-filter or NOT
res = sc3(sce_X, ks = 5, gene_filter = FALSE, biology = FALSE) #ks: number of clusters, need to set ks = true K
label = as.numeric(res$sc3_5_clusters)  #need to set sc3_K_clusters


### Measure the clustering results
K = max(Y) - min(Y) + 1
levels = unique(Y)
perm = combinat::permn(1:K) # list of all permutations
n = length(Y)
error = n
for (p in 1:length(perm)){
  row = perm[[p]]
  true = rep(0,n)
  for (i in 1:K){
    true[Y == levels[i]] = row[i] 
  }
  error = min(error, sum(true != label))
}

###clustering error
print(paste('Clustering error: ', error))
###clustering accuracy
print(paste('Clustering accuracy: ', (n - error)/n))
### ARI
print(paste('ARI: ', ARI(Y, label)))
