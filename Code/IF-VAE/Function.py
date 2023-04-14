##### This file provides functions to implement IF-VAE, VAE and variant IF-VAE(X), VAE(X) in the paper D. Chen, J, Jin, and Z.T. Ke (2023) Subject clustering by IF-PCA and several recent methods.##########
import numpy as np
import pandas as pd
from scipy.stats import norm
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import tensorflow.keras.backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment #minimum weight matching in bipartite graphs
import warnings
warnings.filterwarnings('ignore') #not show any warnings


########### Feature selection (IF-step) ##########
#### Input:
#Data: p-by-n data matrix, n is number of samples, p is number of featuers. Each column presents the observations for a sample. 
#per (optional): a number with 0 < per <= 1, the percentage of Kolmogorov-Smirnov statistics that will be used in the normalization step, default is 1. When the data is highly skewed, this parameter can be specified, such as 0.5.
#### Output:
#ranking: p-by-1 vector shows the ranking for each feature according to ranking with p-values
#numselect: number of selected features in IF-PCA
def feature_selection(Data, per = 1):
    p = len(Data)
    n = len(Data.iloc[0,])

    # Normalize the data
    temp = (Data.T-Data.T.mean())/Data.T.std() #n-by-p
    W = temp.T # p-by-n normalized data matrix W

    # Simulate KS values. this step may takes a while, to reduce time, can replace 50 by a smaller value or use the Matlab code
    rep = 50*p #the number of Kolmogorov-Smirnov statistics to be simulated
    KSvalue = np.zeros(rep)
    for i in range(rep):
        x = np.random.randn(n)
        z = (x - np.mean(x)) / np.std(x)
        z = z / np.sqrt(1 - 1/n)
        pi = np.sort(norm.cdf(z))
        kk = np.arange(n+1) / n
        KSvalue[i] = np.max([np.max(np.abs(kk[0:n] - pi)), np.max(np.abs(kk[1:(n+1)] - pi))])
    KSvalue = KSvalue * np.sqrt(n)

    KSmean = np.mean(KSvalue)
    KSstd = np.std(KSvalue)

    if(per < 1):
        KSvaluesort = np.sort(KSvalue)
        KSmean = np.mean(KSvaluesort[0:round(per*rep)])
        KSstd = np.std(KSvaluesort[0:round(per*rep)])

    # Calculate KS value for each feature in the data set
    KS = np.zeros(p)
    for j in range(p):
        pi = norm.cdf(W.iloc[j,:] / np.sqrt(1 - 1/n))
        pi = np.sort(pi)
        kk = np.arange(n+1) / n
        KS[j] = np.max([np.max(np.abs(kk[0:n] - pi)), np.max(np.abs(kk[1:(n+1)] - pi))])

    if(per == 1):
        KS = (KS - np.mean(KS)) / np.std(KS)
    else:
        KSsort = np.sort(KS)
        KS = (KS - np.mean(KSsort[0:round(per*p)])) / np.std(KSsort[0:round(per*p)])

    # Calculate P-value with simulated KS values
    KSadjust = KS * KSstd + KSmean
    pval = np.zeros(p)
    for i in range(p):
        pval[i] = np.mean(KSvalue > KSadjust[i])

    psort = np.sort(pval)
    ranking = np.argsort(pval)

    # Calculate HC functional at each data point
    kk = np.arange(1,p+1) / (1 + p)
    HCsort = np.sqrt(p) * (kk - psort) / np.sqrt(kk)
    HCsort  = HCsort / np.sqrt(np.maximum(np.sqrt(n) * (kk - psort) / kk, 0) + 1)
    HC = np.zeros(p)
    HC[ranking] = HCsort

    # Decide the threshold
    pvalcut = (np.log(p))/p 
    Ind = np.nonzero(psort > pvalcut)[0][0]
    ratio = HCsort
    ratio[0:Ind] = -np.Inf
    ratio[round(p/2):] = -np.Inf
    L = np.nonzero(ratio == np.max(ratio))[0][-1]
    numselect = L
    
    return[ranking, numselect]


########## Functions and Classes used in VAE
## Function for reparameterization trick to make model differentiable
def sampling(args):   
    import tensorflow as tf
    # Function with args required for Keras Lambda function
    z_mean, z_log_var = args
    # Draw epsilon of the same shape from a standard normal distribution
    epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                              stddev=epsilon_std)
    
    # The latent vector is non-deterministic and differentiable
    # in respect to z_mean and z_log_var
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z

class CustomVariationalLayer(Layer):
    """
    Define a custom layer that learns and performs the training
    This function is borrowed from:
    https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
    """
    def __init__(self, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_input, x_decoded):
        reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - 
                                K.exp(z_log_var_encoded), axis=-1)
        return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

##Implementing Warm-up as described in Sonderby et al. LVAE  
class WarmUpCallback(Callback):
   def __init__(self, beta, kappa):
       self.beta = beta
       self.kappa = kappa
   # Behavior on each epoch
   def on_epoch_end(self, epoch, logs={}):
       if K.get_value(self.beta) <= 1:
           K.set_value(self.beta, K.get_value(self.beta) + self.kappa)   



########### VAE dimensionality reduction ##########
### encoder and decode have one hidden layer, the encoder uses the ReLU activation and the decoder uses the sigmoid activation
#### Input:
#Data: p-by-n data matrix, n is number of samples, p is number of featuers. Each column presents the observations for a sample. 
#### Input(optional):
#latent_dim: dimensionality of the latent space in VAE, default is 25
#batch_size: batch size in training the neural network, default is 50
#epochs: epochs in training the neural network, default is 100
#learning_rate: learning rate in training the neural network, default is 0.0005
#### Output:
# n-by-latent_dim matrix
def VAE(Data,latent_dim = 25, batch_size = 50, epochs = 100, learning_rate = 0.0005):
    np.random.seed(123)
    rnaseq_df = Data.T #n-by-p data matrix
    
    # Split 10% test set randomly
    test_set_percent = 0.1
    rnaseq_test_df = rnaseq_df.sample(frac=test_set_percent)
    rnaseq_train_df = rnaseq_df.drop(rnaseq_test_df.index)

    ###Initialize variables and hyperparameters
    global original_dim
    original_dim = rnaseq_df.shape[1]
    global epsilon_std
    epsilon_std = 1.0
    kappa = 1
    global beta
    beta = K.variable(0)
           
    ### Encoder
    # Input place holder for RNAseq data with specific input size
    rnaseq_input = Input(shape=(original_dim, ))
    
    # Input layer is compressed into a mean and log variance vector of size `latent_dim`
    global z_mean_encoded
    z_mean_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
    z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
    z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)
    
    global z_log_var_encoded
    z_log_var_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
    z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
    z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)
    
    # return the encoded and randomly sampled z vector
    # Takes two keras layers as input to the custom sampling function layer with a `latent_dim` output
    z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean_encoded, z_log_var_encoded])
    
    ### Decoder
    # The decoding layer is much simpler with a single layer and sigmoid activation
    decoder_to_reconstruct = Dense(original_dim, kernel_initializer='glorot_uniform', activation='sigmoid')
    rnaseq_reconstruct = decoder_to_reconstruct(z)
    
    ###Connect the encoder and decoder to make the VAE
    adam = optimizers.Adam(lr=learning_rate)
    vae_layer = CustomVariationalLayer()([rnaseq_input, rnaseq_reconstruct])
    vae = Model(rnaseq_input, vae_layer)
    vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

    ### Train VAE
    vae.fit(np.array(rnaseq_train_df),
                   shuffle=True,
                   epochs=epochs,
                   verbose=0,
                   batch_size=batch_size,
                   validation_data=(np.array(rnaseq_test_df), None)
                   ,callbacks=[WarmUpCallback(beta, kappa),])
    
    ### Encode to the latent space
    encoder = Model(rnaseq_input, z_mean_encoded)
    encoded_rnaseq_df = encoder.predict_on_batch(rnaseq_df)
    encoded_rnaseq_df = pd.DataFrame(encoded_rnaseq_df, index=rnaseq_df.index)
    data_embedding = encoded_rnaseq_df 
    return data_embedding # n-by-latent_dim latent encoding
        
        
    

########### Implement IF-VAE, VAE and variant IF-VAE(X), VAE(X) in one go ##########
#### Input:
#Data: p-by-n data matrix, n is number of samples, p is number of featuers. Each column presents the observations for a sample. 
#num_class: number of classes
#### Input(optional):
#true_label: vector of n, true class labels. If class label is given, can output the number of errors
#feature_selection: boolean, if run IF-step or not, default is True. If run VAE, set feature_selection = False
#normalize: boolean, if VAE is applied to the normalized data matrix or not, defalt is True. If run variant IF-VAE(X) and VAE(X), set normalize = False
#latent_dim: dimensionality of the latent space in VAE, default is 25
#per: in IF-step, a number with 0 < per <= 1, the percentage of Kolmogorov-Smirnov statistics that will be used in the normalization step, default is 1. When the data is highly skewed, this parameter can be specified, such as 0.5.
#batch_size: batch size in training the neural network in VAE, default is 50
#epochs: epochs in training the neural network in VAE, default is 100
#learning_rate: learning rate in training the neural network in VAE, default is 0.0005
#### Output:
#labels: n-by-1 vector, estimated labels for each sample
#error: number of incorrectly clustered sample if true class label is known
def IF_VAE(Data, num_class, true_label = [], feature_select = True, normalize = True, latent_dim = 25, per = 1, batch_size = 50, epochs = 100, learning_rate = 0.0005):
    
    ####### Normalization
    if normalize:
        temp = (Data.T-Data.T.mean())/Data.T.std() #n-by-p
        Data = temp.T # p-by-n normalized data matrix 
  
    ####### Feature Selection
    if feature_select:
        [ranking, numselect] = feature_selection(Data, per = per)
        Datasort = Data.iloc[ranking,:]
        Data = Datasort.iloc[:numselect,:]

    ###### Dimension Reduction by VAE 
    data_embedding = VAE(Data, latent_dim = latent_dim, batch_size = batch_size, epochs = epochs, learning_rate =learning_rate) # n-by-latent_dim
  
    ###### Kmeans clustering
    kmeans = KMeans(init="random", n_clusters=num_class, n_init=10, max_iter=300,random_state=42)
    kmeans.fit(data_embedding) # input n-by-p data matrix
    labels = kmeans.labels_ 

    ##### Number of clustering errors (if class label is known)
    error = 'No clustering errors since true class label is unknown'
    if len(true_label) > 0:
        cm = confusion_matrix(true_label,labels)
        def _make_cost_m(cm):
        	s = np.max(cm)
        	return (- cm + s)
        indexes = linear_assignment(_make_cost_m(cm))
        error = np.sum(cm) -  np.trace(cm[:, indexes[1]]) #number of errors
    
    return[labels, error]

