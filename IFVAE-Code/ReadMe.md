# IF-VAE ReadMe:

This archive contains a Python implementation of IF-VAE, VAE and variant IF-VAE(X), VAE(X), along with an application on 10 Microarray data sets and 8 single-cell data sets as described in the paper *D. Chen, J, Jin, and Z.T. Ke (2023) Subject clustering by IF-PCA and several recent methods*. 

Current version 

V1.1

## SHORT DOCUMENTATION

Contains 3 Python files:

### 1. Function.py: (for the usage of IF-VAE, please see the other 2 example files) 

Provide all functions and classes required to implement IF-VAE. Can implement the IF-VAE and VAE using the main function called ‘IF-VAE’：

Usage: 
```
[labels, error] =  IF_VAE(Data, num_class, true_label = [], feature_select = True, normalize = True, latent_dim = 25, per = 1, batch_size = 50, epochs = 100, learning_rate = 0.0005)
```

**Input**: 
* Data: p-by-n data matrix, n is number of samples, p is number of featuers. Each column presents the observations for a sample. 	
* num_class: number of classes

**Input(optional)**:
* true_label: vector of n, true class labels. If class label is given, can output the number of errors
* feature_selection: boolean, if run IF-step or not, default is True. If run VAE, set feature_selection = False
* normalize: boolean, if VAE is applied to the normalized data matrix or not, defalt is True. If run variant IF-VAE(X) and VAE(X), set normalize = False
* latent_dim: dimensionality of the latent space in VAE, default is 25
* per: in IF-step, a number with 0 < per <= 1, the percentage of Kolmogorov-Smirnov statistics that will be used in the normalization step, default is 1. When the data is highly skewed, this parameter can be specified, such as 0.5.
* batch_size: batch size in training the neural network in VAE, default is 50
* epochs: epochs in training the neural network in VAE, default is 100
* learning_rate: learning rate in training the neural network in VAE, default is 0.0005

**Output**:
* labels: n-by-1 vector, estimated labels for each sample
* error: number of incorrectly clustered sample if true class label is known

For more information about the functions, please see ‘Function.py’ with detailed description. For the usage of IF-VAE function, please see the other 2 example files:

### 2. Example-Microarray.py:

Provide examples to run IF-VAE, VAE, IF-VAE(X), VAE(X) on the 10 Microarray data sets.

### 3. Example-Single-Cell.py:

Provide examples to run IF-VAE, VAE, IF-VAE(X), VAE(X) on the 8 Single-cell data sets.



## LICENSE

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

If you use this code for your publication, please include a reference to the paper ‘D. Chen, J, Jin, and Z.T. Ke (2023) Subject clustering by IF-PCA and several recent methods’.
 
 
## CONTACT
For any problem, please contact
Dieyi Chen
at Harvard University, Department of Statistics.
Email: dieyi.chen@g.harvard.edu

