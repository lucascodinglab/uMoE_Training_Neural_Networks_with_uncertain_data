
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
    warnings.filterwarnings("ignore", category = RuntimeWarning)
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
        
    
    def fit(self, train_data, train_target, valid_data = None, 
            valid_target = None, reg_alpha = 0.5, reg_lambda = 0.0003, 
            lr = 0.001, n_epochs = 100, batch_size_experts = 4, 
            batch_size_gate = 8, local_mode = True, n_samples = 100, 
            threshold_samples = .5, weighted_experts = True, 
            weighted_gate = False, verbose = False, seed = None): 
        self.train_data = train_data
        self.verbose = verbose
        
        # Get Information about Prediction Task
        self.task, input_size, output_size = self.__get_task_type(train_data, np.array(train_target))
        # binary classification
        if self.task == 1:
            loss_fn = nn.BCELoss(reduction = "none") 
        # multi class classification
        elif self.task == 2:
            loss_fn = nn.CrossEntropyLoss(reduction = "none")
        # regression
        elif self.task == 3:
            loss_fn = nn.MSELoss()
        # sampling of KDE and Restriction of samples
        # sampled_data = train_data.sample(n_samples,seed = seed, threshold = threshold_samples)
        sampled_data = train_data.sample(n_samples,seed = seed)

        # clustering
        self.__clustering(self.n_experts, sampled_data)
        
        
        labels_sample = self.pred_clusters(sampled_data)


        # distribution of probability mass after clustering
        prob_dist_train = self.__prob_mass_cluster(self.n_experts, labels_sample,  int(n_samples * threshold_samples))
        
            
        
        # Search the local Mode Value for dominante Cluster and order every instance to the coresponding expert
        if local_mode == True:
             train_data_local =  self.__local_cluster_mode(train_data, prob_dist_train)
             train_loader_list_experts_local = self.__divide_dataset_for_experts(train_data_local, train_target, prob_dist_train, self.n_experts, batch_size = batch_size_experts, weighted_experts = weighted_experts)
        else:
            train_loader_list_experts_global = self.__divide_dataset_for_experts(train_data.mode(), train_target, prob_dist_train, self.n_experts, batch_size = batch_size_experts, weighted_experts = weighted_experts)
        
        # Validation Data 
        if valid_data is not None:
            labels_valid = self.pred_clusters(valid_data.mode())
            prob_dist_valid = self.__prob_mass_cluster(self.n_experts, labels_valid)
        else:
            prob_dist_valid = None
        if valid_data is not None:
            valid_loader_list_experts_global = self.__divide_dataset_for_experts(valid_data.mode(), valid_target, prob_dist_valid, self.n_experts, batch_size = batch_size_experts, weighted_experts = None)
        else:
            valid_loader_list_experts_global = [None] * self.n_experts
            
            
        # Train Experts   
        for i in range(self.n_experts):
            self.experts[i].train_model(train_loader = train_loader_list_experts_local[i], valid_loader = valid_loader_list_experts_global[i],
                                        prob_dist_train = prob_dist_train, n_epochs = n_epochs, loss_fn = loss_fn, weighted_loss = weighted_experts, 
                                        lr = lr, reg_alpha = reg_alpha, reg_lambda = reg_lambda, verbose = self.verbose)   
        
        
        # Train Gate
        
        # Load Global Mode Train und Validation Set for Training of Gate
        train_loader_gate, valid_loader_gate = self.__datasets_for_gate(train_data.mode(), train_target, prob_dist_train, valid_data, valid_target, prob_dist_valid, batch_size = batch_size_gate, weighted_gate = weighted_gate)
        
        print("finished")
        pass
        
    
    def __clustering(self, n_experts, sampled_data):
        """
        Function: Clustering of samples to decompose the input space for the experts
        Input: n_experts (size of cluster), sampled_data (samples of Train Data), Valid_data
        Ouput: labels of Clustering (array)
        """
        cluster_distribution = np.empty(())
        self.kmeans = KMeans(n_init = 10, n_clusters = n_experts, random_state = 1).fit(sampled_data)


    def pred_clusters(self, newdata): 
        return self.kmeans.predict(newdata)




            
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
    
    
    def __local_cluster_mode(self, data, prob_dist):
        """
        Function: Maximize for every Instance the Mode value for the cluster, where the most prob. Mass lies
        Input: train_data, prob_dist(One-Hot_Encoded Prob. Distribution), kmeans (Clustering Instance fit on Samples)
        Output: local_mode_data
        """
        
        data_local_mode = data.mode().copy()
        dominant_cluster = np.argmax(prob_dist, axis=1)
        for i, instance in tqdm(enumerate(data_local_mode), total=len(data_local_mode), desc="Search Local Cluster Mode"):
            missing_dims = data.data[i].indices[1]
            cluster = dominant_cluster[i]
            if len(missing_dims) > 0:
                instance = data_local_mode[i].copy()
                # get kde for instance
                # Calculate Mode value per Cluster
                minimizer_kwargs = {
                    'options': {'maxiter': 50}  # Limit the maximum number of iterations
                }   
                cluster_centers = np.take(self.kmeans.cluster_centers_[cluster], missing_dims)
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
                    
                    return -kde.pdf(inst.reshape(1,-1)) if closest_cluster == cluster else np.inf

                optimize_modal = basinhopping(lambda x: objective_function(x = x, cluster = cluster, kde = data.data[i], instance = instance, missing_dims = missing_dims, centroids = self.kmeans.cluster_centers_), x0=cluster_centers, minimizer_kwargs=minimizer_kwargs, niter=15, stepsize=0.2)                                    
                modal_values = optimize_modal.x.tolist()
                # print(instance)
                for dim, modal_value in zip(missing_dims, modal_values):
                    data_local_mode[i][dim] = modal_value
        return data_local_mode
        
    def __get_task_type(self, X, y):
        """
        Function: get the predictions task type, as well as the output size and the input size
        """
        num_unique_values = len(np.unique(y))
        input_size = X.mode().shape[1]
        # binary classification
        if num_unique_values == 2:
            return 1, input_size, num_unique_values 
        # multiclass classification
        elif num_unique_values > 2 and num_unique_values <= 20:
            return 2, input_size, num_unique_values
        # regression
        else:
            return 3, input_size, 1
        
        
    def __divide_dataset_for_experts(self, X, y, prob_dist, n_experts, batch_size, weighted_experts):
        """
        Function: Divides Dataset for Expert - Every Expert gets Instance, where most of the prob. Mass lies in
        Input: X (data), y(target), prob_dist(Distribution of Prob. Mass after Clustering of Samples), n_experts (number of experts)
        Output: data (list with dataloaders for every expert)
        """
        dominant_cluster = np.argmax(prob_dist, axis=1)
        data = []
        for i in range(n_experts):
            indices = np.where(dominant_cluster == i)
            X_exp = X[indices]
            y_exp = y[indices]
            
            if weighted_experts is False: 
                weights  = np.repeat(1, len(indices)).tolist()
            else: 
                weights_exp = prob_dist[indices]
                weights = np.max(weights_exp, axis=1).tolist()
            dataset_expert = CustomDataset(X_exp, y_exp, weights, self.task)
            loader_expert = DataLoader(dataset_expert, batch_size)
            data.append(loader_expert)
        return data
            

        
    def __datasets_for_gate(self, train_data, train_target, prob_train, valid_data, valid_target, prob_valid, batch_size, weighted_gate):
        """
        Function: Preprocessing Data for Gate Unit
        Output: train and validation data for Gate
        """
        if weighted_gate is False: 
            weights  = np.repeat(1, len(train_target)).tolist()
        else: 
            weights = np.max(prob_train, axis=1).tolist()
        # train
        print(weights)
        train_data_and_dist = np.concatenate((train_data, prob_train), axis=1)
        train_dataset_gate = CustomDataset(train_data_and_dist, train_target, weights, self.task)
        train_loader_gate = DataLoader(train_dataset_gate, batch_size)
        if valid_data is not None:
            # valid
            valid_data_and_dist = np.concatenate((valid_data.mode(), prob_valid), axis=1)
            valid_dataset_gate = CustomDataset(valid_data_and_dist, valid_target, weights, self.task)
            valid_loader_gate = DataLoader(valid_dataset_gate, batch_size)
            return train_loader_gate, valid_loader_gate
        else:
            return train_loader_gate, None
        
        
    def _init_model(self):
    
        output_experts = self.inputsize if not self.probs_to_gate else self.inputsize + self.n_experts
        self.experts = [Custom_nn(inputs = self.inputsize,  outputs = output_experts, dropout = self.dropout) for i in range(self.n_experts)]
        self.gate = Custom_nn( inputs = output_experts * self.n_experts, outputs = self.outputsize, dropout = self.dropout)
        
    
    
class CustomDataset(Dataset):
    """
    Custom Dataset Class, which is able to process X, y and the weights of the instances
    """
    def __init__(self, X, y, weights = None, task = None):
        
        #to torch tensors
        self.X = torch.tensor(X, dtype=torch.float64)
        if task == 1 or 2:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float64)
            
        #if None all = 1
        
        if weights == None: 
            self.weights = torch.Tensor.repeat(torch.Tensor([1]), len(self.y), dtype=torch.float64)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float64)
        
        
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.weights[index]
    
    

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
    def __init__(self, inputs, outputs, hidden=[64,64], activation=nn.ReLU(), dropout=0, task = None ):
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

        self.task = 'numeric' if task is None else task


    def forward(self, x):
        if self.task == 'numeric': 
            return self.stacked_layers(x)
        
        if self.task == 'binary': 
            return torch.sigmoid(self.stacked_layers(x))
        
        if self.task == 'multiclass': 
            return torch.nn.Softmax(self.stacked_layers(x))

    # def l1_loss(self, w):
    #     return torch.abs(w).sum()
    
    # def l2_loss(self, w):
    #     return torch.square(w).sum()
    
    def train_model(self, train_loader, valid_loader, prob_dist_train, n_epochs, loss_fn, weighted_loss, lr, reg_alpha, reg_lambda, verbose):
        best_score = np.inf
        best_weights = None
        history = []
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        for epoch in range(n_epochs):
            self.train()
            for i, data in enumerate(train_loader):
                
                X_batch, y_batch, weights_batch = data
                # Convert input data to the same data type as the model's weights
                y_batch = y_batch.to(torch.float64)  
                 
                optimizer.zero_grad()

                y_pred = self(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss = torch.mean(loss * weights_batch)
                
                # Elastic Net regularization (L1 + L2)
                l1_regularization = torch.tensor(0., dtype=torch.float64)
                for param in self.parameters():
                    l1_regularization += torch.norm(param, p=1)
    
                l2_regularization = torch.tensor(0., dtype=torch.float64)
                for param in self.parameters():
                    l2_regularization += torch.norm(param, p=2)
    
                loss += reg_lambda * (reg_alpha * l1_regularization + (1 - reg_alpha) * l2_regularization) 

                
                loss.backward()
                optimizer.step()

                       
        
            # Validation is optional
            loss_val = None
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
    
    
                    loss_val = loss_fn(y_preds, y_targets)
                    loss_val = float(loss_val.numpy())
                    history.append(loss_val)
                    if loss_val < best_score:
                        best_score = loss_val
                        best_weights = copy.deepcopy(self.state_dict())
            if (verbose == True) and (epoch % 5 == 0):
                print(f"Epoch: {epoch}, Train_Loss: {loss}, Val_Loss: {loss_val}")
        if valid_loader is not None:
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
    X = uf.uframe_from_array_mice_2(X_true, kernel = "gaussian" , p =.5, mice_iterations = 2)
    # X.analysis(X_true, save= "filename", bins = 20)
    

    
    moe = MoE(4, inputsize=8, outputsize=1)
    moe.fit(X, y, threshold_samples=1, local_mode = True, weighted_experts=True, verbose=True)
    

    
