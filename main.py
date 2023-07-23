
import uframe as uf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from scipy.optimize import basinhopping
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset
import copy
import warnings


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

    # Ignore RuntimeWarnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
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
        
    
    def fit(self, train_data, train_target, valid_data = None, valid_target = None, reg_alpha = 0.5, reg_lambda = 0.0003, lr = 0.001, n_epochs = 100, batch_size_experts = 4, batch_size_gate = 8, local_mode = True, n_samples = 100, threshold_samples = .5, weighted_experts = True, weighted_gate = False, verbose = False ): 
        self.train_data = train_data

        # sampling of KDE and Restriction of samples
        sampled_data = train_data.sample(n_samples,seed = 1)
        sampled_data = self.__threshold_sampling(sampled_data, n_samples, threshold_samples)
        
        # clustering
        labels_sample, labels_valid, kmeans = self.__clustering(self.n_experts, sampled_data, valid_data)
        
        # distribution of probability mass after clustering
        threshold_samples_count = int(n_samples * threshold_samples)
        prob_dist_train = self.__prob_mass_cluster(self.n_experts, labels_sample, threshold_samples_count)
        if labels_valid is not None:
            prob_dist_valid = self.__prob_mass_cluster(self.n_experts, labels_valid)
            
        
        # Search the local Mode Value for dominante Cluster and order every instance to the coresponding expert
        if local_mode == True:
             train_data_local =  self.__local_cluster_mode(train_data, prob_dist_train, kmeans)
             train_loader_list_experts_local = self.___divide_dataset_for_experts(train_data_local, train_target, prob_dist_train, self.n_experts, batch_size = batch_size_experts)
        else:
            train_loader_list_experts_global = self.___divide_dataset_for_experts(train_data.mode(), train_target, prob_dist_train, self.n_experts, batch_size = batch_size_experts)
        
        # Divide Validation Data for experts
        if valid_data is not None:
            valid_loader_list_experts_global = self.___divide_dataset_for_experts(valid_data.mode(), valid_target, prob_dist_valid, self.n_experts, batch_size = batch_size_experts)
        else:
            valid_loader_experts = None
            

        # Get Information about Prediction Task
        task, input_size, output_size = self.__get_task_type(train_data, np.array(train_target))
        # binary classification
        if task == 1:
              loss_fn = nn.BCELoss() 
        # multi class classification
        elif task == 2:
            loss_fn = nn.CrossEntropyLoss()
        # regression
        elif task == 3:
            loss_fn = nn.MSELoss()
            
        # Train Experts   
        
        for expert in range(self.n_experts):
            experts = Custom_nn(input_size, output_size)
            optimizer = optim.Adam(experts.parameters(), lr=lr)
            train_loader_expert = train_loader_list_experts_local[expert]
            experts.train(train_loader_expert, valid_loader_experts, prob_dist_train, n_epochs, optimizer, loss_fn, weighted_experts)   
            
        
        
        # Train Gate
        
        # Load Global Mode Train und Validation Set for Training of Gate
        train_dataset_gate_global = CustomDataset(train_data.mode(), train_target)
        train_loader_gate_global = DataLoader(train_dataset_gate_global, batch_size = batch_size_gate)
        if valid_data is not None:
            valid_dataset_gate = CustomDataset(valid_data.mode(), valid_target)
            valid_loader_gate = DataLoader(valid_dataset_gate, batch_size = batch_size_gate)
        else:
            valid_loader_gate = None
        
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
        kmeans = KMeans(n_clusters = n_experts, random_state = 1).fit(sampled_data)
        labels_sample = kmeans.labels_
        if valid_data is not None:
            labels_valid = kmeans.predict(valid_data.mode())
            return labels_sample, labels_valid
        else:
            return labels_sample, None, kmeans
            
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
    
    
    def __local_cluster_mode(self, data, prob_dist, kmeans):
        """
        Function: Maximize for every Instance the Mode value for the cluster, where the most prob. Mass lies
        Input: train_data, prob_dist(One-Hot_Encoded Prob. Distribution), kmeans (Clustering Instance fit on Samples)
        Output: local_mode_data
        """
        n_clusters = prob_dist.shape[1]
        centroids = kmeans.cluster_centers_
        data_local_mode = data.mode().copy()
        dominant_cluster = np.argmax(prob_dist, axis=1)
        for i, instance in tqdm(enumerate(data_local_mode), total=len(data_local_mode), desc="Search Local Cluster Mode"):
            missing_dims = data.data[i].indices[1]
            cluster = dominant_cluster[i]
            if len(missing_dims) > 0:
                instance = data_local_mode[i].copy()
                # get kde for instance
                kde = data.data[i].continuous
                # Calculate Mode value per Cluster
                minimizer_kwargs = {
                    'options': {'maxiter': 50}  # Limit the maximum number of iterations
                }   
                cluster_centers = np.take(kmeans.cluster_centers_[cluster], missing_dims)
                # print(cluster_centers)
                def objective_function(x, cluster, kde, instance, missing_dims, centroids):
                    """
                    Function: Moves the Gradient to the Max. of the KDE under the Restriction, 
                    that the instance with the local mode lies in the cluster, where most prob. mass lies
                    """
                    inst = instance.copy()
                    x = np.array(x)  
                    inst[missing_dims] = x
                    inst = inst.reshape(1, -1)
                    # Distance to all Centroids
                    distances = cdist(centroids, inst)
                    # Index of closest Centroid
                    closest_cluster = np.argmin(distances)
                    # Find the closest point and get its cluster label
                    
                    return -kde.score_samples(x.reshape(1,-1)) if closest_cluster == cluster else np.inf

                optimize_modal = basinhopping(lambda x: objective_function(x, cluster, kde, instance, missing_dims, centroids), x0=cluster_centers, minimizer_kwargs=minimizer_kwargs, niter=15, stepsize=0.2)                                    
                modal_values = optimize_modal.x.tolist()
                # print(instance)
                for i, modal_value in zip(missing_dims, modal_values):
                    instance[i] = modal_value
                data_local_mode[i] = instance
          
        return data_local_mode
        
    def __get_task_type(self, X, y):
        """
        Function: get the predictions task type, as well as the output size and the input size
        """
        num_unique_values = len(np.unique(y))
        input_size = X.data.mode().shape[1]
        # binary classification
        if num_unique_values == 2:
            return 1, input_size, num_unique_values 
        # multiclass classification
        elif num_unique_values > 2 and num_unique_values <= 20:
            return 2, input_size, num_unique_values
        # regression
        else:
            return 3, input_size, 1
        
        
        
        
        
    def __divide_dataset_for_experts(self, X, y, prob_dist, n_experts, batch_size):
        """
        Function: Divides Dataset for Expert - Every Expert gets Instance, where most of the prob. Mass lies in
        Input: X (data), y(target), prob_dist(Distribution of Prob. Mass after Clustering of Samples), n_experts (number of experts)
        Output: data (list with dataloaders for every expert)
        """
        dominant_cluster = np.argmax(prob_dist, axis=1)
        data = []
        for i in range(n_experts):
            indices = np.where(dominant_cluster == i)
            X = X[indices]
            y = y[indices]
            dataset_expert = CustomDataset(X, y)
            loader_expert = DataLoader(dataset_expert, batch_size)
            data.append(loader_expert)
        return data
            

            
            
               
               
               
        
        
            
    def _init_model(self):
    
        output_experts = self.inputsize if not self.probs_to_gate else self.inputsize + self.n_experts
        self.experts = [Custom_nn(inputs = self.inputsize,  outputs = output_experts, dropout = self.dropout) for i in range(self.n_experts)]
        self.gate = Custom_nn( inputs = output_experts * self.n_experts, outputs = self.outputsize, dropout = self.dropout)
        
    
    
class CustomDataset(Dataset):
    """
    Custom Dataset Class, which is able to process U-Frame instances
    """
    def __init__(self, X, y):
        data = np.concatenate((X,y),axis=1)
        self.data = data
    def __len__(self, data):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    
    

class Custom_nn(nn.Module):
    
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
        super(Custom_nn, self).__init__()

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
    
    def train(self, train_loader, valid_loader, prob_dist_train, n_epochs, optimizer, loss_fn, weighted_loss):
        best_score = np.inf
        best_weights = None
        history = []
    
        for epoch in range(n_epochs):
            self.train()
            with tqdm(train_loader, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for X_batch, y_batch in bar:
                    optimizer.zero_grad()
    
                    # X_batch und y_batch werden bereits als Tensoren geliefert
                    y_pred = self(X_batch)
                    loss = loss_fn(y_pred, y_batch)
    
                    l2_loss = self.l2_loss()  # Annahme: self.l2_loss() ist eine Methode, die den L2-Verlust berechnet
                    loss += l2_loss
                    
                    if weighted_loss:
                        loss = torch.mean(loss * weights_batch)
                    loss.backward()
                    optimizer.step()
    
                    bar.set_postfix(mse=float(loss))
                        
        
            # Validation is optional
            if valid_loader is not None:
                self.eval()
                with torch.no_grad():
                    y_preds = []
                    y_targets = []
                    for X_val, y_val in valid_loader:
                        y_pred = self(X_val)
                        y_preds.append(y_pred)
                        y_targets.append(y_val)
    
                    y_preds = torch.cat(y_preds, dim=0)
                    y_targets = torch.cat(y_targets, dim=0)
    
                    mse = mean_squared_error(y_preds.cpu().numpy(), y_targets.cpu().numpy())
                    mse = float(mse)
                    if epoch % 5 == 0:
                        print(mse)
                    history.append(mse)
                    if mse < best_score:
                        best_sore = mse
                        best_weights = copy.deepcopy(self.state_dict())
    
        self.load_state_dict(best_weights)
        return history, best_score
        

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
    y = StandardScaler().fit_transform(data.target[:50].reshape(-1,1))
    

    #uncertainty in data 
    X = uf.uframe_from_array_mice(X_true, kernel = "gaussian" , p =.5, mice_iterations = 2)
    X.analysis(X_true, save= "filename", bins = 20)
    

    
    moe = MoE(4)
    moe.fit(X, y, threshold_samples=.5, local_mode = True)
    
