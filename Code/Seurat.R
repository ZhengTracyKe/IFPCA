#### This file is run on RStudio Cloud under R 4.0.5 (Seurat needs R version > 4.0)
#### Code reference: https://rpubs.com/aomidsal/682508
#### Seurat don't input fix number of clusters, check number of cluster in 'FindClusters' function
#### the result is stable with fixed tuning parameters, one iteration is enough
### different tuning parameters may give slightly different errors
library('Seurat')
library("Matrix")
library('aricode')#ARI


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


##### Run Seurat
sobj = CreateSeuratObject(counts = X, project = 'SeuratProject', min.cells = 10, min.features = 20) 
sobj = FindVariableFeatures(sobj, selection.method = "vst", nfeatures =1000) #select top-'nfeatures' features
sobj = ScaleData(sobj, features = rownames(sobj)) #center and scale data
sobj = RunPCA(sobj, features = VariableFeatures(object = sobj), npcs = 50) #get first 50 PCs
sobj1 = FindNeighbors(sobj, k.param = 20) #KNN
#should tune 'resolution' below to make 'number of communities' = true K
sobj2 = FindClusters(sobj1, resolution = 0.6) #low 'resolution' leads to smaller number of clusters
label = as.numeric(Idents(sobj2)) 


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

