# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:50:28 2023

@author: lucas
"""



import uframe as uf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import torch.nn as nn
import torch



class MoE(): 
    """
    
    A class used for storing and working with uncertain data.

    ...

    ----------
    data : list
        A list of data instances of class uframe_instance

    Methods
    -------
    fit()
        Trains the Moe 
        
    predict(n = 1, seed = None)
        predicts with the the model 

    """

    
    def __init__(self, n_experts, hidden_experts = [64,64], hidden_gate = [64,64], dropout = 0, inputsize = 1, outputsize = 1, probs_to_gate = True) : 
    
        self.n_experts = n_experts 
        self.hidden_experts = hidden_experts
        self.hidden_gate = hidden_gate
        self.dropout = dropout 
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.probs_to_gate = probs_to_gate
        
        self._init_model()
        
        
        
    def predict(self, new_data): 
        
        pass
        
    
    def fit(self, train_data, train_target, valid_data = None, valid_target = None, reg_alpha = 0.5, reg_lambda = 0.0003, lr = 0.001, local_mode = True, n_samples = 100, threshold_samples = .5, weighted_experts = True, weighted_gate = False, verbose = False ): 
        self.train_data = train_data

        # sampling of KDE and Restriction of samples
        sampled_data = train_data.sample(n_samples)
        sampled_data = self.__threshold_sampling(sampled_data, n_samples, threshold_samples)
        
        # clustering
        labels_sample, labels_valid = self.__clustering(self.n_experts, sampled_data, valid_data)
        print(type(labels_sample))
        
        # distribution of probability mass after clustering
        prob_dist_train = self.__prob_mass_cluster(self.n_experts, labels_sample, n_samples)
        if labels_valid is not None:
            prob_dist_valid = self.__prob_mass_cluster(self.n_experts, labels_valid)
        print(prob_dist_train) 
        
        # Search the local Mode Value for dominante Cluster
        if local_mode == True:
            data = self.__local_cluster_mode(train_data, prob_dist_train)
            
        
        #Mode 
        train_data.mode()


        train_data.ev()        
        
        #train experts 
        
        
        #train gate 
        
        
        pass
        
    def __threshold_sampling(self, sampled_data, n_samples, threshold_samples):
        if threshold_samples <1: 
            ind=[]
            n_choosen = round(n_samples * threshold_samples)
            
            for i in range(len(self.train_data)): 
                pdfs = self.train_data.data[i].pdf(sampled_data[i*n_samples:(i+1)*n_samples,:])
                sort_ind = np.argsort(pdfs, axis = 0)

                sort_ind = np.squeeze(sort_ind[:])
                sort_ind = sort_ind[sort_ind < n_choosen]

                ind.append(sort_ind+(i*n_samples))
                
            sampled_data = sampled_data[np.concatenate(ind, axis = 0 ),]                
                
        return sampled_data
    
    def __clustering(self, n_experts, sampled_data, valid_data):
        """
        Function: Clustering of samples to decompose the input space for the experts
        Input: n_experts (size of cluster), sampled_data (samples of Train Data), Valid_data
        Ouput: labels of Clustering (array)
        """
        cluster_distribution = np.empty(())
        kmeans = KMeans(n_clusters = n_experts).fit(sampled_data)
        labels_sample = kmeans.labels_
        if valid_data is not None:
            labels_valid = kmeans.predict(valid_data.mode())
            return labels_sample, labels_valid
        else:
            return labels_sample, None
            
    def __prob_mass_cluster(self, n_experts, labels, n_samples = 1):
        """
        Function: Get Probability Distribution across Clusters
        Input: n_experts, labels (train,val,test), n_samples
        Output: prob_dist (Distribution of Probability Mass across Clusters in One-Hot-Encoded (ndarray))
        """
        num_sections = len(labels) // n_samples
        prob_dist = np.zeros((num_sections, n_experts))
        
        for i in range(num_sections):
            section = labels[i * n_samples : (i + 1) * n_samples]
            unique_cluster, counts = np.unique(section, return_counts=True)
            relative_frequency = counts / len(section)
            prob_dist[i][unique_cluster - 1] = relative_frequency
        return prob_dist
    
    
    def __local_cluster_mode(self, train_data, prob_dist):
        """
        Function: Maximize for every Instance the Mode value, where the most prob. Mass lies
        Input: train_data, prob_dist(One-Hot_Encoded Prob. Distribution)
        Output: train_data (with new mode values)
        """
        pass
        
            
    def _init_model(self):
    
        output_experts = self.inputsize if not self.probs_to_gate else self.inputsize + self.n_experts
        self.experts = [ custom_nn(inputs = self.inputsize,  outputs = output_experts, dropout = self.dropout) for i in range(self.n_experts)]
        self.gate = custom_nn( inputs = output_experts * self.n_experts, outputs = self.outputsize, dropout = self.dropout)
        
    
    



class custom_nn(nn.Module):
    
    r"""

    Parameters
    ----------
    inputs : int
        number of input nodes 
    outputs : int
        number of output nodes 
    hidden : list, optional
        list of nodes in hidden lazers. The default is [30, 30, 30].
    activation : torch fct, optional
        troch function that is used as activation in all layers. The default is nn.ReLU().
    dropout : bool/float, optional
        If not False, dropout is set to specified percentage . The default is False.
 
    Returns
        creates class 
   

    """
    def __init__(self, inputs, outputs, hidden=[64,64], activation=nn.ReLU(), dropout=0):
        r"""

        Parameters
        ----------
        inputs : int
            number of input nodes 
        outputs : int
            number of output nodes 
        hidden : list, optional
            list of nodes in hidden lazers. The default is [30, 30, 30].
        activation : torch fct, optional
            troch function that is used as activation in all layers. The default is nn.ReLU().
        dropout : float, optional
           dropout is set to specified percentage. The default is 0.

        Returns
            creates class 
        

        """
        super(custom_nn, self).__init__()

        torch.set_default_dtype(torch.float64)

        layer_list = list()
        layer_list.append(nn.Linear(inputs, hidden[1]))
        layer_list.append(activation)
        for i in range(1, len(hidden)):
            layer_list.append(nn.Linear(hidden[i-1], hidden[i]))
            layer_list.append(activation)
            layer_list.append(nn.Dropout(dropout))

        layer_list.append(nn.Linear(hidden[-1], outputs))

        self.stacked_layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.stacked_layers(x)

def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=1)
        nn.init.zeros_(module.bias)

def init_uniform(module):
    if type(module) == nn.Linear:
        nn.init.uniform_(module.weight, a=-1.0, b=1.0)
        nn.init.zeros_(module.bias)




def eval_moe(**kwargs): 
    pass
    
    
    pass

if __name__ == "__main__":
    
    #data loading and preprocessing: 
    data = fetch_california_housing()


    #scaling 

    X_true = MinMaxScaler().fit_transform(data.data[:50,:])
    y = StandardScaler().fit_transform(data.target.reshape(-1,1))



    #uncertainty in data 
    X = uf.uframe_from_array_mice(X_true, kernel = "gaussian" , p =.5, mice_iterations = 2)
    X.analysis(X_true, save= "filename", bins = 20)
    
    moe = MoE(4)
    moe.fit(X,y,threshold_samples=0.5)
    
    



    
    