############ Examples to implement IF-VAE, VAE, IF-VAE(X), VAE(X) in 8 Single-Cell data sets describe in the paper D. Chen, J, Jin, and Z.T. Ke (2023) Subject clustering by IF-PCA and several recent methods. ######
import pandas as pd
import Function as F
path = '/Users/name/Desktop/Data/Single-Cell/' # path of the data sets

########### import 8 Single-Cell RNA-seq Data Sets ###########
### Name set: (load p-by-n data matrix)
#camp1, 13111-by-777
#camp2, 11233-by-734
#darmanis, 13400-by-466
#deng, 16347-by-268
#goolam, 21199-by-124
#grun, 5547-by-1502
#li, 25369-by-561
#patel, 5948-by-430
name = 'camp1'

Data = pd.read_csv(path + name + '-x-filter.txt', sep=",", header = None) # p-by-n data matrix after the pre-processing
labels_true = pd.read_csv(path + name + '-y.txt', sep=" ", header=None).iloc[:,0] # n-by-1 true class labels, start from 0
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
print('Clustering accuracy: ', 1 - error/Data.shape[1])


########### Only run feature selection (IF-step)
#[ranking, numselect] = F.feature_selection(Data = Data) # input p-by-n data matrix
#ranking: p-by-1 ranking of featurs
#numselect: number of influential features selected 

