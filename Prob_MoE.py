import uframe as uf
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, accuracy_score
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
    def __init__(self, n_experts, hidden_experts = [64,64], hidden_gate = [64,64], dropout = 0, inputsize = 1, outputsize = 1, probs_to_gate = True) : 
    
        self.n_experts = n_experts 
        self.hidden_experts = hidden_experts
        self.hidden_gate = hidden_gate
        self.dropout = dropout 
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.probs_to_gate = probs_to_gate
        
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
        
        # cluster test data
        labels_test = self.pred_clusters(test_data)
        prob_dist_test = self.__prob_mass_cluster(self.n_experts, labels_test, n_samples = 1)
        
        # load data for expert and gate
        test_dataset_expert = CustomDataset(test_data)
        test_loader_expert = DataLoader(test_dataset_expert, batch_size = 10)
        test_data_gate = np.concatenate((test_data, prob_dist_test), axis=1)
        test_dataset_gate = CustomDataset(test_data_gate)
        test_loader_gate = DataLoader(test_dataset_gate, batch_size = 10)
        
        # iterate through dataloader
        predictions = []
        for data_gate, data_expert in zip(test_loader_gate, test_loader_expert):
            X_batch_gate, y_batch_gate, weights_batch_gate = data_gate
            X_batch_expert, y_batch_expert, weights_batch_expert = data_expert
            
            predictions.extend(self.gate.forward(X_batch_gate, X_batch_expert).detach().numpy())

        return predictions
        
        
        
        
    
    def fit(self, train_data, train_target, valid_data = None, 
            valid_target = None, reg_alpha = 0.5, reg_lambda = 0.0003, 
            lr = 0.001, n_epochs = 100, batch_size_experts = 4, 
            batch_size_gate = 8, local_mode = True, n_samples = 100, 
            threshold_samples = .5, weighted_experts = True, 
            weighted_gate = False, verbose = False, seed = None): 
        """
        Train the Mixture of Experts (MoE) model using the provided training data and target values.
        
        Parameters:
        -----------
        train_data : ndarray
            The input training data to be used for training the MoE model. The shape of the array should be (n_samples, n_features).
        train_target : ndarray
            The target values corresponding to the training data. The shape of the array should be (n_samples,) for classification tasks, or (n_samples, 1) for regression tasks.
        valid_data : ndarray, optional
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
        local_mode : bool, optional
            A boolean flag indicating whether to use local mode for training. If True, the training data will be divided based on the dominant cluster of each instance. If False, global mode will be used, where all instances are used for training the experts. Default is True.
        n_samples : int, optional
            The number of samples to be used for training the MoE model in the case of local mode. Default is 100.
        threshold_samples : float, optional
            The threshold for selecting instances for training in the case of local mode. Only instances whose cluster probability is above this threshold will be used for training the experts. Default is 0.5.
        weighted_experts : bool, optional
            A boolean flag indicating whether to apply weights to the expert loss during training. If True, the losses of individual experts will be weighted based on the probability distribution of clusters. Default is True.
        weighted_gate : bool, optional
            A boolean flag indicating whether to apply weights to the gate loss during training. If True, the gate loss will be weighted based on the probability distribution of clusters. Default is False.
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
        
        1. Preprocess the input data and determine the prediction task type (binary classification, multi-class classification, or regression).
        2. Sample data using Kernel Density Estimation (KDE) and restrict samples to maintain a manageable size.
        3. Perform clustering on the sampled data to determine expert assignments for each instance.
        4. Distribute the probability mass of clusters based on clustering results.
        5. If local mode is enabled, use the dominant cluster to order instances for each corresponding expert. Otherwise, use global mode and directly divide the data for expert training.
        6. Optionally, use validation data for monitoring the model's performance during training.
        7. Train each expert model with the specified loss function, regularization, and optimization settings.
        8. Train the gate model to combine the outputs of individual experts and perform predictions.
        
        The function optimizes the parameters of the MoE model to best fit the training data and task type while considering the provided hyperparameters and settings.
        
        """
        
        self.train_data = train_data
        self.verbose = verbose
        self.batch_size_gate = batch_size_gate
        self.local_mode = local_mode
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
            loss_fn = nn.MSELoss(reduction = "none")
        # sampling of KDE and Restriction of samples
        sampled_data = train_data.sample(n_samples,seed = seed, threshold = threshold_samples)
        # sampled_data = train_data.sample(n_samples,seed = seed)

        # clustering
        self.__clustering(self.n_experts, sampled_data)
        
        
        labels_sample = self.pred_clusters(sampled_data)


        # distribution of probability mass after clustering
        self.prob_dist_train = self.__prob_mass_cluster(self.n_experts, labels_sample,  int(n_samples * threshold_samples))
        
            
        
        # Search the local Mode Value for dominante Cluster and order every instance to the coresponding expert
        if local_mode == True:
            # local mode
             self.train_data_local =  self.__local_cluster_mode(train_data, self.prob_dist_train)
             train_loader_list_experts = self.__divide_dataset_for_experts(self.train_data_local, train_target, self.prob_dist_train, self.n_experts, batch_size = batch_size_experts, weighted_experts = weighted_experts)
        else:
            # global mode
            train_loader_list_experts = self.__divide_dataset_for_experts(train_data.mode(), train_target, self.prob_dist_train, self.n_experts, batch_size = batch_size_experts, weighted_experts = weighted_experts)
        
        # Validation Data 
        if valid_data is not None:
            labels_valid = self.pred_clusters(valid_data.mode())
            prob_dist_valid = self.__prob_mass_cluster(self.n_experts, labels_valid)
            valid_loader_list_experts_global = self.__divide_dataset_for_experts(valid_data.mode(), valid_target, prob_dist_valid, self.n_experts, batch_size = batch_size_experts, weighted_experts = None)
        else:
            prob_dist_valid = None
            valid_loader_list_experts_global = [None] * self.n_experts
            
            
        # Train Experts  
            
        for i in range(self.n_experts):
            self.experts[i].task = self.task
            self.experts[i].train_model(train_loader = train_loader_list_experts[i], valid_loader = valid_loader_list_experts_global[i],
                                        n_epochs = n_epochs, loss_fn = loss_fn, weighted_loss = weighted_experts, lr = lr, 
                                        reg_alpha = reg_alpha, reg_lambda = reg_lambda, verbose = self.verbose)   
        
        
        # Train Gate
        
        # Load Global Mode Train und Validation Set for Training of Gate
        train_loader_expert, train_loader_gate, valid_loader_expert, valid_loader_gate = self.__datasets_for_gate(train_data.mode(), train_target, self.prob_dist_train, valid_data, valid_target,
                                                                                                                  prob_dist_valid, batch_size = self.batch_size_gate, weighted_gate = weighted_gate)
        

        self.gate.train_model(train_loader_expert, train_loader_gate, valid_loader_expert, valid_loader_gate, n_epochs, loss_fn, weighted_gate, lr, reg_alpha, reg_lambda, verbose)

    
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
                
                with warnings.catch_warnings():
                    # deactivate warning about optimizer np.inf error
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
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
                # weights  = np.repeat(1, len(indices)).tolist()
                weights = None
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
        # train gate
        train_data_and_dist = np.concatenate((train_data, prob_train), axis=1)
        train_dataset_gate = CustomDataset(train_data_and_dist, train_target, weights, self.task)
        train_loader_gate = DataLoader(train_dataset_gate, batch_size)
        # train expert
        train_dataset_expert = CustomDataset(train_data, train_target, weights, self.task)
        train_loader_expert = DataLoader(train_dataset_expert, batch_size)
        if valid_data is not None:
            # valid gate
            valid_data_and_dist = np.concatenate((valid_data.mode(), prob_valid), axis=1)
            valid_dataset_gate = CustomDataset(valid_data_and_dist, valid_target, weights = None, task = self.task)
            valid_loader_gate = DataLoader(valid_dataset_gate, batch_size)
            # valid expert
            valid_dataset_expert = CustomDataset(valid_data.mode(), valid_target, weights = None, task = self.task)
            valid_loader_expert = DataLoader(valid_dataset_expert, batch_size)
            return train_loader_expert, train_loader_gate, valid_loader_expert, valid_loader_gate
        else:
            return train_loader_expert, train_loader_gate, None, None
        
        
    def _init_model(self):
    
        self.experts = [Custom_nn(inputs = self.inputsize,  outputs = self.outputsize, dropout = self.dropout) for i in range(self.n_experts)]
        self.gate = Gate_nn(inputs = self.inputsize + self.n_experts, outputs = self.n_experts, 
                            dropout = self.dropout, trained_experts_list = self.experts)
   
        
    def evaluation(self, predictions, true_targets):
        if self.task in (1,2):
            score = accuracy_score(true_targets, np.round(predictions, 0)) * 100
        else:
            score = mean_squared_error(true_targets, predictions)
        
        return score
    
    def analyze(self, data_certain, save_path):
        self.save_path = save_path + str("_analysis.pdf")
        labels_certain = self.pred_clusters(data_certain)
        prob_dist_certain = self.__prob_mass_cluster(self.n_experts, labels_certain)
            
        dominant_clusters_local, dominant_clusters_global = self.__analyze_clustering(prob_dist_certain)
        if self.local_mode:
            self.__cluster_change(dominant_clusters_local, dominant_clusters_global)
    
    def __analyze_clustering(self, prob_dist_certain):
        num_clusters = len(prob_dist_certain[0])  # Number of clusters
        cluster_accuracies_global = np.zeros(num_clusters)
        dominant_clusters_certain = np.argmax(prob_dist_certain, axis=1)
        
        # global clustering
        labels_global_mode = self.pred_clusters(self.train_data.mode())
        prob_dist_global_mode = self.__prob_mass_cluster(self.n_experts, labels_global_mode, n_samples = 1)
        dominant_clusters_global = np.argmax(prob_dist_global_mode, axis=1)
                
        if self.local_mode:
            dominant_clusters_local = self.pred_clusters(self.train_data_local)
            # prob_dist_local_mode = self.__prob_mass_cluster(self.n_experts, labels_local_mode, n_samples = 1)
            # dominant_clusters_local = np.argmax(prob_dist_local_mode, axis=1)
            cluster_accuracies_local = np.zeros(num_clusters)
        
        for cluster in range(num_clusters):
            cluster_indices = np.where(dominant_clusters_certain == cluster)[0]
            correct_predictions_global = np.sum(dominant_clusters_global[cluster_indices] == cluster)
            cluster_accuracy_global = correct_predictions_global / len(cluster_indices)
            cluster_accuracies_global[cluster] = cluster_accuracy_global
            
            if self.local_mode:
               correct_predictions_local = np.sum(dominant_clusters_local[cluster_indices] == cluster)
               cluster_accuracy_local = correct_predictions_local / len(cluster_indices)
               cluster_accuracies_local[cluster] = cluster_accuracy_local
    
        # Calculate the weighted average of cluster accuracies for global and local modes
        weighted_average_global = accuracy_score(dominant_clusters_certain, dominant_clusters_global)
        if self.local_mode:
            weighted_average_local = accuracy_score(dominant_clusters_certain, dominant_clusters_local)
    
        # Save the plots to the PDF file
        with PdfPages(self.save_path) as pdf:
            # Plot the cluster accuracies for the global mode as a bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(range(num_clusters), cluster_accuracies_global, color='blue', alpha=0.7)
            plt.axhline(y=weighted_average_global, color='r', linestyle='--', label='Weighted Average (Global)')
            plt.xlabel('Cluster')
            plt.ylabel('Accuracy')
            plt.title(f'Global - Cluster Assignment Accuracy (Average: {weighted_average_global:.2f})')
            plt.legend()
            plt.xticks(range(num_clusters))
            
            # Save the global mode plot to PDF
            pdf.savefig()
            plt.close()
        
            if self.local_mode:
                # Plot the cluster accuracies for the local mode as a bar chart
                plt.figure(figsize=(10, 6))
                plt.bar(range(num_clusters), cluster_accuracies_local, color='green', alpha=0.7)
                plt.axhline(y=weighted_average_local, color='r', linestyle='--', label='Weighted Average (Local)')
                plt.xlabel('Cluster')
                plt.ylabel('Accuracy')
                plt.title(f'Local - Cluster Assignment Accuracy (Average: {weighted_average_local:.2f})')
                plt.legend()
                plt.xticks(range(num_clusters))
                
                # Save the local mode plot to PDF
                pdf.savefig()
                plt.close()
        if self.local_mode:
            return dominant_clusters_local, dominant_clusters_global
        else:
            return None, dominant_clusters_global
                
    def __cluster_change(self, old, new):
         transitions = {}
         for oldone, newone in zip(old, new):
             key = f"{oldone} -> {newone}"
             transitions[key] = transitions.get(key, 0) + 1
         
         nodes = sorted(list(set(old + new)))
         
         # Create a dictionary to store the migration counts for each original class
         migration_counts = {node: [transitions.get(f"{node} -> {n}", 0) for n in nodes] for node in nodes}
         
         # Create the Grouped Bar chart
         grouped_bar = go.Figure()
         for i, node in enumerate(nodes):
             grouped_bar.add_trace(go.Bar(
                 x=[f"{node} -> {n}" for n in nodes],
                 y=migration_counts[node],
                 name=f"Original Class {node}",
             ))
         
         # Anpassung der Layout-Eigenschaften
         grouped_bar.update_layout(
             title='Migration between Clusters',
             xaxis_title="Changes",
             yaxis_title="Count",
             font=dict(size=10),
         )
         
         grouped_bar.show()
         
         output_file = self.save_path + "migration_diagram_grouped_bar.html"
         pyo.plot(grouped_bar, filename=output_file, auto_open=False)
        
class CustomDataset(Dataset):
    """
    Custom Dataset Class, which is able to process X, y and the weights of the instances
    """
    def __init__(self, X, y = None, weights = None, task = None):
        
        #to torch tensors
        self.X = torch.tensor(X, dtype=torch.float64)
        if task in (1, 2, 3):
            self.y = torch.tensor(y, dtype=torch.float64)
        else:
            self.y = torch.tensor(np.zeros(len(X)), dtype=torch.float64)
        
        #if None all = 1
        
        if weights == None: 
            self.weights = torch.tensor(np.ones(len(X)), dtype=torch.float64)
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



    def forward(self, x):
        # binary
        if self.task == 1:
            return nn.Sigmoid()(self.stacked_layers(x))
        # multiclass
        if self.task == 2: 
            return torch.argmax(nn.Softmax(dim=1)(self.stacked_layers(x)), dim=1).to(torch.float64).unsqueeze(1)
        # regression
        if self.task == 3: 
            return self.stacked_layers(x)

    def train_model(self, train_loader, valid_loader, n_epochs, loss_fn, weighted_loss, lr, reg_alpha, reg_lambda, verbose):
        best_score = np.inf
        best_weights = None
        history = []
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        for epoch in range(n_epochs):
            self.train()
            for i, data in enumerate(train_loader):
                
                X_batch, y_batch, weights_batch = data
                # Convert input data to the same data type as the model's weights
                 
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = loss_fn(y_pred, y_batch.unsqueeze(1))
                loss = torch.mean(loss * weights_batch)
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
                    for X_val, y_val, weights_val in valid_loader:
                        y_pred_val = self(X_val)
                        y_preds.append(y_pred_val)
                        y_targets.append(y_val)
    
                    y_preds = torch.cat(y_preds, dim=0)
                    y_targets = torch.cat(y_targets, dim=0)
    
                    # print(f" {y_preds.shape} (y_preds), {y_targets.unsqueeze(1).shape} (y_targets)")
                    loss_val = loss_fn(y_preds, y_targets.unsqueeze(1))
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
   

class Gate_nn(Custom_nn):
    """
    Subclass of Custom_nn used for training of the Gate Unit
    """
    def __init__(self, inputs, outputs, hidden=[64, 64], activation=nn.ReLU(), dropout=0, trained_experts_list = None, task = 3):
        super(Gate_nn, self).__init__(inputs, outputs, hidden, activation, dropout)
        self.task = task
        self.trained_experts_list = nn.ModuleList(trained_experts_list)

    def forward(self, inputs_gate, inputs_expert):
        # Propagate inputs_gate through the model
        softmax_layer = nn.Softmax(dim=-1)
        gate_output = softmax_layer(super(Gate_nn, self).forward(inputs_gate))
        expert_outputs_weighted = 0

        expert_outputs = []
        expert_weights = []
        for i, expert in enumerate(self.trained_experts_list):
            expert_output = expert(inputs_expert)
            expert_output.detach()  # mark expert weights as not trainable
            expert_outputs_weighted += expert_output * gate_output[:, i:i+1]
            expert_outputs.append(expert_output)
            expert_weights.append(gate_output[:, i:i+1])

        output = expert_outputs_weighted.sum(dim=1)
        
        return output
    
    def train_model(self, train_loader_expert, train_loader_gate, valid_loader_expert, valid_loader_gate, n_epochs, loss_fn, weighted_loss, lr, reg_alpha, reg_lambda, verbose):
        best_score = np.inf
        best_weights = None
        history = []
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        for epoch in range(n_epochs):
            self.train()
            for i, (data_expert, data_gate) in enumerate(zip(train_loader_expert, train_loader_gate)):
                X_batch_gate, y_batch_gate, weights_batch_gate = data_gate
                X_batch_expert, y_batch_expert, weights_batch_expert = data_expert
                # Convert input data to the same data type as the model's weights
                y_batch_gate = y_batch_gate.to(torch.float64)  
                 
                optimizer.zero_grad()

                y_pred = self(X_batch_gate, X_batch_expert)
                loss = loss_fn(y_pred, y_batch_gate)
                loss = torch.mean(loss * weights_batch_gate)
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
                        X_val_expert, y_val_expert, weights_val_expert = valid_data_expert
                        X_val_gate, y_val_gate, weights_val_gate = valid_data_gate
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


    
    	


if __name__ == "__main__":
    
    pass