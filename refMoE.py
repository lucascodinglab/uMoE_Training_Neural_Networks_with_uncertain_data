import uframe as uf
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from tqdm import tqdm
from scipy.optimize import basinhopping
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset
import copy
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.offline as pyo
import os


class MoE(): 
    """
    A class for training and using the Mixture of Experts (MoE) model for uncertain data analysis.

    Attributes:
    ----------
    data : list
        A list of data instances of class `uframe_instance`.

    Methods:
    -------
    fit(train_data, train_target, valid_data=None, valid_target=None, reg_alpha=0.5, reg_lambda=0.0003,
        lr=0.001, n_epochs=100, batch_size_experts=4, batch_size_gate=8, local_mode=True, n_samples=100,
        threshold_samples=0.5, weighted_experts=True, weighted_gate=False, verbose=False, seed=None)
        Trains the MoE model on the provided training data and targets. It allows for specifying validation data
        and various hyperparameters for training customization.

    predict(test_data)
        Predicts the output using the trained Mixture of Experts (MoE) model for uncertain data.

    Description:
    -----------
    The `MoE` class is designed to handle uncertain data analysis using the Mixture of Experts (MoE) model.
    It provides functionalities for training the MoE model on uncertain data and making predictions.

    The main method, `fit()`, allows users to train the MoE model on the given training data and targets.
    Users can provide optional validation data to monitor the model's performance during training. The `fit()`
    method also allows users to customize various hyperparameters, such as the learning rate, number of epochs,
    batch sizes, regularization parameters, and more.

    Once the MoE model is trained, the `predict()` method enables users to make predictions on new uncertain data.
    The method clusters the test data based on the trained MoE model's clustering algorithm and then loads the
    data into the expert and gate models to make predictions. The final predictions are stored in a list.

    The MoE model is particularly useful when dealing with uncertain data, where each instance has a probability
    distribution over its possible values. It can handle classification tasks, regression tasks, and binary
    classification problems with ease.

    The `MoE` class is an essential tool for researchers and practitioners working with uncertain data and seeking
    to leverage the power of the Mixture of Experts model for accurate predictions and meaningful uncertainty
    estimates.
    """
    def __init__(self, n_experts, inputsize, outputsize, hidden_experts = [64,64], hidden_gate = [64,64], dropout = 0): 
    
        self.n_experts = n_experts 
        self.hidden_experts = hidden_experts
        self.hidden_gate = hidden_gate
        self.dropout = dropout 
        self.inputsize = inputsize
        self.outputsize = outputsize
        self._init_model()
        
        
    def predict(self, test_data):
        """
        Predicts the output using the trained Mixture of Experts (MoE) model.
    
        Parameters
        ----------
        test_data : ndarray
            MinMax scaled data for testing. The shape of the array should be (n_samples, n_features).
    
        Returns
        -------
        list
        A list containing the predicted outputs from the Gate for each data point in the 'test_data'.
    
        Notes
        -----
        This function clusters the test data based on the trained MoE model's clustering algorithm.
        Then, it loads the test data into the expert and gate models to make predictions.
        The final predictions are stored in the 'predictions' list.
        """
        
        
        # load data for expert and gate
        test_dataset_expert = CustomDataset(test_data)
        test_loader_expert = DataLoader(test_dataset_expert, batch_size = 10)
        test_dataset_gate = CustomDataset(test_data)
        test_loader_gate = DataLoader(test_dataset_gate, batch_size = 10)
        
        # iterate through dataloader
        predictions = []
        for data_gate, data_expert in zip(test_loader_gate, test_loader_expert):
            X_batch_gate, y_batch_gate = data_gate
            X_batch_expert, y_batch_expert = data_expert
            
            predictions.extend(self.gate.forward(X_batch_gate, X_batch_expert).detach().numpy())

        return predictions
        
        
    
    
    def fit(self, train_data, train_target, valid_data = None, 
            valid_target = None, reg_alpha = 0.5, reg_lambda = 0.0003, 
            lr = 0.001, n_epochs = 100, batch_size_experts = 4, 
            batch_size_gate = 8, verbose = False, seed = None): 
        """
        Train the Mixture of Experts (MoE) model using the provided training data and target values.
        
        Parameters:
        -----------
        train_data : Uframe Object
            The input training data to be used for training the MoE model. The shape of the array should be (n_samples, n_features).
        train_target : ndarray
            The target values corresponding to the training data. The shape of the array should be (n_samples,) for classification tasks, or (n_samples, 1) for regression tasks.
        valid_data : Uframe Object, optional
            The optional validation data to be used for monitoring the model's performance during training. The shape of the array should be (n_samples, n_features). Default is None.
        valid_target : ndarray, optional
            The optional target values corresponding to the validation data. The shape of the array should be (n_samples,) for classification tasks, or (n_samples, 1) for regression tasks. Default is None.
        reg_alpha : float, optional
            The weight parameter for the Elastic Net regularization L1 term. Default is 0.5.
        reg_lambda : float, optional
            The weight parameter for the Elastic Net regularization L2 term. Default is 0.0003.
        lr : float, optional
            The learning rate for the optimizer used during training. Default is 0.001.
        n_epochs : int, optional
            The number of epochs to train the MoE model. Default is 100.
        batch_size_experts : int, optional
            The batch size for training the individual expert models. Default is 4.
        batch_size_gate : int, optional
            The batch size for training the gate model. Default is 8.           
        verbose : bool, optional
            A boolean flag indicating whether to print progress and training information during the training process. Default is False.
        seed : int, optional
            The seed value for random number generation. Default is None.

        Returns:
        --------
        None
        
        Notes:
        ------
        This function performs the following steps to train the MoE model:
        
        The function optimizes the parameters of the MoE model to best fit the training data and task type while considering the provided hyperparameters and settings.
        
        """
        init_normal(self.gate)
        init_normal(self.experts)
        self.train_data = train_data
        self.verbose = verbose
        self.batch_size_gate = batch_size_gate
        
        # Get Information about Prediction Task
        self.task, input_size, output_size = self.__get_task_type(train_data, np.array(train_target))
        # Check if the task is multi-class classification (task == 2) and validate target format
        if self.task == 2:
            if train_target.ndim <= 1:
                raise ValueError("For multi-class classification (task == 2), train_target must be one-hot encoded.")
            if valid_target is not None and valid_target.ndim <= 1:
                raise ValueError("For multi-class classification (task == 2), valid_target must be one-hot encoded.")

        # binary classification
        if self.task == 1:
            loss_fn = nn.BCELoss() 
        # multi class classification
        elif self.task == 2:
            loss_fn = nn.CrossEntropyLoss()
        # regression
        elif self.task == 3:
            loss_fn = nn.MSELoss()
        # clustering
        self.__clustering(self.n_experts, train_data.mode())
        
        
        labels_train = self.pred_clusters(train_data.mode())


        train_loader_list_experts = self.__divide_dataset_for_experts(train_data.mode(), train_target, labels_train, self.n_experts, batch_size = batch_size_experts)
        
        # Validation Data 
        if valid_data is not None:
            labels_valid = self.pred_clusters(valid_data.mode())
            valid_loader_list_experts_global = self.__divide_dataset_for_experts(valid_data.mode(), valid_target, labels_valid, self.n_experts, batch_size = batch_size_experts)
        else:
            valid_loader_list_experts_global = [None] * self.n_experts
            
            
        # Train Experts  
            
        for i in range(self.n_experts):
            self.experts[i].task = self.task
            self.experts[i].train_model(train_loader = train_loader_list_experts[i], valid_loader = valid_loader_list_experts_global[i],
                                        n_epochs = n_epochs, loss_fn = loss_fn, lr = lr, 
                                        reg_alpha = reg_alpha, reg_lambda = reg_lambda, verbose = self.verbose)
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
        
        # Load Global Mode Train und Validation Set for Training of Gate
        train_loader_expert, train_loader_gate, valid_loader_expert, valid_loader_gate = self.__datasets_for_gate(train_data.mode(), train_target, valid_data, valid_target, batch_size = self.batch_size_gate)
        

        self.history, self.bestScore = self.gate.train_model(train_loader_expert, train_loader_gate, valid_loader_expert, valid_loader_gate, n_epochs, loss_fn, lr, reg_alpha, reg_lambda, verbose, self.task)


    def __clustering(self, n_experts, sampled_data):
        """
        Function: Clustering of samples to decompose the input space for the experts
        Input: n_experts (size of cluster), sampled_data (samples of Train Data), Valid_data
        Ouput: labels of Clustering (array)
        """
        os.environ["OMP_NUM_THREADS"] = "2" #KMeans
        self.kmeans = KMeans(n_init = 10, n_clusters = n_experts, random_state = 31).fit(sampled_data)


    def pred_clusters(self, newdata): 
        return self.kmeans.predict(newdata)

            
    def __prob_mass_cluster(self, n_experts, labels, n_samples=1):
        """
        Function: Get Probability Distribution across Clusters
        Input: n_experts, labels (train, val, test), n_samples
        Output: prob_dist (Distribution of Probability Mass across Clusters in One-Hot-Encoded (ndarray))
        """
        num_sections = len(labels) // n_samples
        prob_dist = np.zeros((num_sections, n_experts))
    
        for i in range(num_sections):
            section = labels[i * n_samples : (i + 1) * n_samples]
            unique_cluster, counts = np.unique(section, return_counts=True)
            relative_frequency = counts / len(section)
    
            # Initialize the prob_dist row with the correct ordering of relative frequencies
            for j, cluster_id in enumerate(range(n_experts)):
                if cluster_id in unique_cluster:
                    prob_dist[i][j] = relative_frequency[np.where(unique_cluster == cluster_id)][0]
    
        return prob_dist
        
    
   
        
    def __get_task_type(self, X, y):
        """
        Function: get the predictions task type, as well as the output size and the input size
        """
        num_unique_values = len(np.unique(y))
        input_size = X.mode().shape[1]
        # binary classification
        if num_unique_values == 2 and self.outputsize == 1:
            return 1, input_size, num_unique_values 
        # multiclass classification
        elif self.outputsize > 1:
            return 2, input_size, self.outputsize
        # regression
        else:
            return 3, input_size, 1
        
        
    def __divide_dataset_for_experts(self, X, y, dominant_cluster, n_experts, batch_size):
        """
        Function: Divides Dataset for Expert - Every Expert gets Instance, where most of the prob. Mass lies in
        Input: X (data), y(target), prob_dist(Distribution of Prob. Mass after Clustering of Samples), n_experts (number of experts)
        Output: data (list with dataloaders for every expert)
        """
        data = []
        for i in range(n_experts):
            indices = np.where(dominant_cluster == i)
            X_exp = X[indices]
            y_exp = y[indices]
            
            dataset_expert = CustomDataset(X_exp, y_exp, self.task)
            loader_expert = DataLoader(dataset_expert, batch_size)
            data.append(loader_expert)
        return data
            

        
    def __datasets_for_gate(self, train_data, train_target, valid_data, valid_target, batch_size):
        """
        Function: Preprocessing Data for Gate Unit
        Output: train and validation data for Gate
        """

        # train gate
        train_dataset_gate = CustomDataset(train_data, train_target,  self.task)
        train_loader_gate = DataLoader(train_dataset_gate, batch_size)
        # train expert
        train_dataset_expert = CustomDataset(train_data, train_target, self.task)
        train_loader_expert = DataLoader(train_dataset_expert, batch_size)
        if valid_data is not None:
            # valid gate
            valid_dataset_gate = CustomDataset(valid_data.mode(), valid_target,task = self.task)
            valid_loader_gate = DataLoader(valid_dataset_gate, batch_size)
            # valid expert
            valid_dataset_expert = CustomDataset(valid_data.mode(), valid_target, task = self.task)
            valid_loader_expert = DataLoader(valid_dataset_expert, batch_size)
            return train_loader_expert, train_loader_gate, valid_loader_expert, valid_loader_gate
        else:
            return train_loader_expert, train_loader_gate, None, None
        
        
    def _init_model(self):
    
        self.experts = [Custom_nn(inputs = self.inputsize,  outputs = self.outputsize, dropout = self.dropout) for i in range(self.n_experts)]
        self.gate = Gate_nn(inputs = self.inputsize, outputs = self.n_experts, 
                            dropout = self.dropout, trained_experts_list = self.experts)
   
        
    def evaluation(self, predictions, true_targets):
        """
        Evaluates the performance of the predictions against the true targets based on the model's task.
    
        Parameters
        ----------
        predictions : array-like
            Predicted values obtained from the model.
        true_targets : array-like
            True target values (ground truth).
    
        Returns
        -------
        score : float
            The evaluation score representing the performance of the predictions.
    
        Notes
        -----
        The method evaluates the performance of the model's predictions against the true targets
        based on the task type of the model.
        """
        if self.task in (1,2):
            score = accuracy_score(true_targets, np.round(predictions, 0)) * 100
        else:
            score = mean_squared_error(true_targets, predictions)
        
        return score
    
    def analyze(self, data_certain):
        """
        Analyze the clustering results and generate plots.
    
        Parameters
        ----------
        data_certain : np.array
            Data used for certain prediction.
        save_path : str
            Location where the analysis plots will be saved (including the folder path).
    
        Returns
        -------
    
        """
        
        labels_certain = self.pred_clusters(data_certain)
            
        cluster_accuracies_global, silhouette = self.__analyze_clustering(labels_certain)
        return cluster_accuracies_global, self.bestScore
    def __analyze_clustering(self, dominant_clusters_certain):
        num_clusters = self.n_experts  # Number of clusters
        cluster_accuracies_global = np.zeros(num_clusters)

        # global clustering
        dominant_clusters_global = self.pred_clusters(self.train_data.mode())

        
        for cluster in range(num_clusters):
            cluster_indices = np.where(dominant_clusters_certain == cluster)[0]
            correct_predictions_global = np.sum(dominant_clusters_global[cluster_indices] == cluster)
            cluster_accuracy_global = correct_predictions_global / len(cluster_indices)
            cluster_accuracies_global[cluster] = cluster_accuracy_global
            
    
        # Calculate the weighted average of cluster accuracies for global and local modes
        weighted_average_global = accuracy_score(dominant_clusters_certain, dominant_clusters_global)

        #silhouette score
        silhouette = silhouette_score(self.train_data.mode(), dominant_clusters_global)
        return weighted_average_global, silhouette
                
        
class CustomDataset(Dataset):
    """
    Custom Dataset Class, which is able to process X, y and the weights of the instances
    """
    def __init__(self, X, y = None, task = None):
        
        #to torch tensors
        self.X = torch.tensor(X, dtype=torch.float64)
        if task in (1, 2, 3):
            self.y = torch.tensor(y, dtype=torch.float64)
        else:
            self.y = torch.tensor(np.zeros(len(X)), dtype=torch.float64)      
            
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    

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
    def __init__(self, inputs, outputs, hidden=[16,16], activation=nn.ReLU(), dropout=0.3, task = None ):
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
        # binary
        if self.task == 1:
            return nn.Sigmoid()(self.stacked_layers(x))
        # multiclass
        if self.task == 2: 
            return nn.Softmax(dim=1)(self.stacked_layers(x))
        # regression
        if self.task == 3: 
            return self.stacked_layers(x)

    def train_model(self, train_loader, valid_loader, n_epochs, loss_fn, lr, reg_alpha, reg_lambda, verbose):
        best_score = np.inf
        best_weights = None
        history = []
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        for epoch in range(n_epochs):
            self.train()
            for i, data in enumerate(train_loader):
                
                X_batch, y_batch = data
                # Convert input data to the same data type as the model's weights
                 
                optimizer.zero_grad()
                y_pred = self(X_batch)
                # print(f"y_pred {y_pred}, y_pred.shape {y_pred.shape},\n y_batch {y_batch}, y_batch.shape {y_batch.shape}")
                if self.task in (1,3): # 1-D
                    loss = torch.mean(loss_fn(y_pred, y_batch.unsqueeze(1)))
                else: # n-D
                    loss = torch.mean(loss_fn(y_pred, y_batch))
                loss_print = loss.clone()
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
                        y_pred_val = self(X_val)
                        y_preds.append(y_pred_val)
                        y_targets.append(y_val)
    
                    y_preds = torch.cat(y_preds, dim=0)
                    y_targets = torch.cat(y_targets, dim=0)
    
                    # print(f" {y_preds.shape} (y_preds), {y_targets.unsqueeze(1).shape} (y_targets)")
                    if self.task in (1,3):
                        loss_val = loss_fn(y_preds, y_targets.unsqueeze(1))
                    else:
                        loss_val = loss_fn(y_preds, y_targets)
                    loss_val = float(loss_val.mean())
                    history.append(loss_val)
                    if loss_val < best_score:
                        best_score = loss_val
                        best_weights = copy.deepcopy(self.state_dict())
            if (verbose == True) and (epoch % 5 == 0):
                print(f"Epoch: {epoch}, Train_Loss: {loss_print}, Val_Loss: {loss_val}")
        if valid_loader is not None:
            self.load_state_dict(best_weights)
        return history, best_score
   

class Gate_nn(nn.Module):
    """
    Subclass of Custom_nn used for training of the Gate Unit
    """
    def __init__(self, inputs, outputs, hidden=[16, 16], activation=nn.ReLU(), dropout=0, trained_experts_list=None):
        super(Gate_nn, self).__init__()

        self.trained_experts_list = nn.ModuleList(trained_experts_list)
        # Create the neural network layers
        layers = []
        input_size = inputs
        for units in hidden:
            layers.append(nn.Linear(input_size, units))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = units
        layers.append(nn.Linear(input_size, outputs))
        
        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)
        
    def forward(self, inputs_gate, inputs_expert):
        # Propagate inputs_gate through the model
        softmax_layer = nn.Softmax(dim=-1)
        gate_output = softmax_layer(self.model(inputs_gate))
        expert_outputs_weighted = 0

        expert_outputs = []
        expert_weights = []
        for i, expert in enumerate(self.trained_experts_list):
            expert_output = expert(inputs_expert)
            expert_output.detach()  # mark expert weights as not trainable
            expert_outputs_weighted += expert_output * gate_output[:, i:i+1]
            expert_outputs.append(expert_output)
            expert_weights.append(gate_output[:, i:i+1])
        
        if self.task in (1,3):
            output = expert_outputs_weighted.sum(dim=1)
        else:
            norm_weights = torch.sum(expert_outputs_weighted, dim=1, keepdim=True)
            output = expert_outputs_weighted / norm_weights
            
        return output
    
    def train_model(self, train_loader_expert, train_loader_gate, valid_loader_expert, valid_loader_gate, n_epochs, loss_fn, lr, reg_alpha, reg_lambda, verbose, task):
        self.task = task
        best_score = np.inf
        best_weights = None
        history = []
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        for epoch in range(n_epochs):
            self.train()
            for i, (data_expert, data_gate) in enumerate(zip(train_loader_expert, train_loader_gate)):
                X_batch_gate, y_batch_gate = data_gate
                X_batch_expert, y_batch_expert = data_expert
                # Convert input data to the same data type as the model's weights
                y_batch_gate = y_batch_gate.to(torch.float64)  
                 
                optimizer.zero_grad()

                y_pred = self(X_batch_gate, X_batch_expert)
                loss = loss_fn(y_pred, y_batch_gate)
                loss_print = loss.clone()
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
            if valid_loader_gate is not None:
                self.eval()
                with torch.no_grad():
                    y_preds = []
                    y_targets = []
                    for (valid_data_expert),(valid_data_gate) in zip(valid_loader_expert, valid_loader_gate):
                        X_val_expert, y_val_expert = valid_data_expert
                        X_val_gate, y_val_gate = valid_data_gate
                        y_pred = self(X_val_gate, X_val_expert)
                        y_preds.append(y_pred)
                        y_targets.append(y_val_gate)
    
                    y_preds = torch.cat(y_preds, dim=0)
                    y_targets = torch.cat(y_targets, dim=0)
                    # print(f"{y_preds.shape} (y_preds), {y_targets.unsqueeze(1).shape} (y_targets)")
                    loss_val = loss_fn(y_preds, y_targets)
                    loss_val = float(loss_val.mean())
                    history.append(loss_val)
                    if loss_val < best_score:
                        best_score = loss_val
                        best_weights = copy.deepcopy(self.state_dict())
            if (verbose == True) and (epoch % 5 == 0):
                print(f"Epoch: {epoch}, Train_Loss: {loss_print}, Val_Loss: {loss_val}")
        if valid_loader_gate is not None:
            self.load_state_dict(best_weights)
        return history, best_score


    
 
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=1)
        nn.init.zeros_(module.bias)   	


if __name__ == "__main__":
    pass

    

        
    
    
  