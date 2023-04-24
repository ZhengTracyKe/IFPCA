############ Examples to implement IF-VAE, VAE, IF-VAE(X), VAE(X) in 10 Microarray data sets described in the paper D. Chen, J, Jin, and Z.T. Ke (2023) Subject clustering by IF-PCA and several recent methods.######
import pandas as pd
import Function as F
from sklearn.metrics.cluster import adjusted_rand_score #comoute ARI
path = '/Users/name/Desktop/Data/Microarray/' # path of the data sets

########### import 10 Microarray Data Sets ###########
# brain
Data = pd.read_csv(path + "brain.x.txt", sep=" ", header=None) # p-by-n data matrix
labels_true = pd.read_csv(path + "brain.y.txt", sep=" ", header=None).T.values[0] # n-by-1 true class labels, start from 0
num = max(labels_true) - min(labels_true) + 1 #number of classes

#breast
Data = pd.read_csv(path + "breast.x.txt", sep=",", header=None) # p-by-n data matrix
labels_true = pd.read_csv(path + "breast.y.txt", sep=" ", header=None).iloc[:,3] # n-by-1 true class labels, start from 0
num = max(labels_true) - min(labels_true) + 1 #number of classes

# colon
Data = pd.read_csv(path + "colon.x.txt", sep=" ", header=None) # p-by-n data matrix
labels_true = pd.read_csv(path + "colon.y.txt", sep=" ", header=None).T.values[0] # n-by-1 true class labels, start from 0
num = max(labels_true) - min(labels_true) + 1 #number of classes

# leukemia
Data = pd.read_csv(path + "leukemia.x.txt", sep=" ", header=None) # p-by-n data matrix
labels_true = pd.read_csv(path + "leukemia.y.txt", sep=" ", header=None).T.values[0] # n-by-1 true class labels, start from 0
num = max(labels_true) - min(labels_true) + 1 #number of classes

# lung1
Data = pd.read_csv(path + "lung1.x.txt", sep=",", header=None) # p-by-n data matrix
labels_true = pd.read_csv(path + "lung1.y.txt", sep=" ", header=None).iloc[:,3]# n-by-1 true class labels, start from 0
num = max(labels_true) - min(labels_true) + 1 #number of classes

# lung2
Data = pd.read_csv(path + "lung2.x.txt", sep=",", header=None) # p-by-n data matrix
labels_true = pd.read_csv(path + "lung2.y.txt", sep=",", header=None).T.values[0]# n-by-1 true class labels, start from 0
num = max(labels_true) - min(labels_true) + 1 #number of classes

# lymphoma
Data = pd.read_csv(path + "lymphoma.x.txt", sep=" ", header=None) # p-by-n data matrix
labels_true = pd.read_csv(path + "lymphoma.y.txt", sep=" ", header=None).T.values[0] # n-by-1 true class labels, start from 0
num = max(labels_true) - min(labels_true) + 1 #number of classes

# prostate
Data = pd.read_csv(path + "prostate.x.txt", sep=" ", header=None) # p-by-n data matrix
labels_true = pd.read_csv(path + "prostate.y.txt", sep=" ", header=None).T.values[0] # n-by-1 true class labels, start from 0
num = max(labels_true) - min(labels_true) + 1 #number of classes

# srbct
Data = pd.read_csv(path + "srbct.x.txt", sep=" ", header=None) # p-by-n data matrix
labels_true = pd.read_csv(path + "srbct.y.txt", sep=" ", header=None).T.values[0] # n-by-1 true class labels, start from 0
num = max(labels_true) - min(labels_true) + 1 #number of classes

# sucancer
Data = pd.read_csv(path + "su.x.txt", sep=",", header=None) # p-by-n data matrix
labels_true = pd.read_csv(path + "su.y.txt", sep=" ", header=None).iloc[:,3]# n-by-1 true class labels, start from 0
num = max(labels_true) - min(labels_true) + 1 #number of classes



############ Run IF-VAE, IF-VAE(X), VAE, VAE(X) with or without class lables #################
##### run IF_VAE, with true class label
[labels, error] = F.IF_VAE(Data = Data, num_class = num, true_label = labels_true, latent_dim = 25)
##### run IF_VAE(X), with true class label
#[labels, error] = F.IF_VAE(Data = Data, num_class = num, true_label = labels_true, normalize = False)
##### run VAE, with true class label
#[labels, error] = F.IF_VAE(Data = Data, num_class = num, true_label = labels_true, feature_select = False)
##### run VAE(X), with true class label
#[labels, error] = F.IF_VAE(Data = Data, num_class = num, true_label = labels_true, feature_select = False, normalize = False)
##### run IF_VAE, without true class label
#[labels, error] = F.IF_VAE(Data = Data, num_class = num)

### print the result
print('Predicted class labels: ', labels)
print('Number of errors: ', error)
print('ARI: ', adjusted_rand_score(labels_true,labels))

########### Only run feature selection (IF-step)
#[ranking, numselect] = F.feature_selection(Data = Data) # input p-by-n data matrix
#ranking: p-by-1 ranking of featurs
#numselect: number of influential features selected 

