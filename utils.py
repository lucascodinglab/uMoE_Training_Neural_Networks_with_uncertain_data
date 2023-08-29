import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import csv
import os


def preprocess_data(data_path = None, dataset = None):
    """
    Preprocesses the specified dataset and returns the preprocessed data along with input and output sizes.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset to be preprocessed. Possible values are 'diabetes', 'breast_cancer', 'california',
        'blood', 'energy', 'water_potability', 'wine_quality', and 'microbes'. The default is None.

    Returns
    -------
    X : ndarray or DataFrame
        The input features of the preprocessed data.
    y : ndarray
        The target labels of the preprocessed data (in case of classification tasks) or the target values (in case of regression tasks).
    input_size : int
        The number of input features (i.e., the number of columns in X).
    output_size : int
        The number of output classes (in case of classification tasks) or the number of target values (in case of regression tasks).
    score_type : str
        Type of Task (MSE/Accuracy)
    Notes
    -----
    The function reads the specified dataset from the provided data path and performs the following preprocessing steps:

    - 'diabetes': Reads the diabetes dataset from a CSV file, separates the input features (X) and target labels (y).
    - 'breast_cancer': Reads the breast cancer dataset from a CSV file, removes the 'id' column, encodes target labels 'M' as 0 and 'B' as 1.
    - 'california': Fetches the California housing dataset from scikit-learn, shuffles the data, and separates input features and target values.
    - 'blood': Reads the blood donation dataset from a CSV file, separates input features and target labels.
    - 'energy': Reads the energy dataset from a CSV file, removes the 'Y2' column, and separates input features and target values.
    - 'water_potability': Reads the water potability dataset from a CSV file, removes rows with missing values, separates input features and target labels.
    - 'wine_quality': Reads the wine quality dataset from a CSV file, separates input features and target values, and transforms target values by subtracting 3.
    - 'microbes': Reads the microbes dataset from a CSV file, removes the 'Unnamed: 0' column, encodes categorical variables, separates input features and target labels.
    - 'wifi localization': Reads wifi dataset from .txt file 
    If the dataset name is not provided or not recognized, the function returns None for all outputs.

    """
    data_path = data_path
    if dataset == "diabetes":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv") 
        X = data.drop(columns='Outcome').values
        y = np.array(data["Outcome"].reset_index(drop=True))
        input_size = X.shape[1]
        output_size = 1
        score_type = "Accuracy"
    elif dataset == "breast_cancer":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv") 
        data.drop(columns=["id"],axis=1,inplace=True)
        X = data.drop(columns='diagnosis').values
        y = data["diagnosis"]
        y = y.reset_index(drop=True)
        y = np.array(y.map({'M': 0, 'B': 1}))
        input_size = X.shape[1]
        output_size = 1
        score_type = "Accuracy"
    elif dataset == "california":
        data = fetch_california_housing()
        X = data.data
        y = data.target
        input_size = X.shape[1]
        output_size = 1
        score_type = "MSE"
    elif dataset == "blood":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv")
        X = data.drop(columns='whether he/she donated blood in March 2007').values
        y = np.array(data["whether he/she donated blood in March 2007"])
        input_size = X.shape[1]
        output_size = 1
        score_type = "Accuracy"
    elif dataset == "energy":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv")
        data.drop("Y2",axis=1,inplace=True)
        X = data.drop(columns='Y1').values
        y = np.array(data["Y1"])
        input_size = X.shape[1]
        output_size = 1
        score_type = "MSE"
    elif dataset == "water_potability":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv")
        data.dropna(inplace=True)
        X = data.drop(columns='Potability').values
        y = np.array(data["Potability"])
        input_size = X.shape[1]
        output_size = 1  
        score_type = "Accuracy"
    elif dataset == "wine_quality":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv")
        X = data.drop(columns='quality').values
        y = data["quality"]
        y = np.array(np.subtract(y, 3))
        input_size = X.shape[1]
        output_size = 6
        score_type = "Accuracy"
        label_encoder = LabelEncoder()
        y_labels = label_encoder.fit_transform(y)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit_transform(y_labels.reshape(-1, 1))
    elif dataset == "microbes":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv")
        data.drop(columns=["Unnamed: 0"],axis=1,inplace =True)
        str_cols = data.select_dtypes(include=['object']).columns
        data[str_cols] = data[str_cols].apply(lambda x: pd.Categorical(x).codes)
        X = data.drop(columns='microorganisms')
        X = X.values
        y = np.array(data["microorganisms"])
        input_size = X.shape[1]
        output_size = 10
        score_type = "Accuracy"     
    elif dataset == "wifi":
        data = pd.read_csv(data_path + "\\" + dataset + ".txt", delimiter='\t', header=None)
        X = data.drop(columns=7).values
        y = np.array(data[7])
        input_size = X.shape[1]
        output_size = 4
        score_type = "Accuracy"
    elif dataset == "banana":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv")
        X = data.drop(columns="Class", axis=1).values
        y = np.array(data["Class"].map({-1: 0, 1:1}))
        input_size = X.shape[1]
        output_size = 1
        score_type = "Accuracy"  
    elif dataset == "life":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv")        
        data.drop(columns=["Year", "Country","Status"],axis=1,inplace=True)
        data.dropna(inplace=True)
        # Delete Data (over all Attributes)
        X = data.drop(columns='Life expectancy ')
        X = X.values
        y = np.array(data["Life expectancy "])
        input_size = X.shape[1]
        output_size = 1
        score_type = "MSE"
    
    elif dataset == "artificial":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv")        
        # Delete Data (over all Attributes)
        X = data.drop(columns='target')
        X = X.values
        y = np.array(data["target"])
        input_size = X.shape[1]
        output_size = 3
        score_type = "Accuracy"
        
    if output_size > 1:
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder()
        y_labels = label_encoder.fit_transform(y)
        y = onehot_encoder.fit_transform(y_labels.reshape(-1, 1)).toarray()
        
    return X, y, input_size, output_size, score_type



def compare_scores(cluster_accuracies_local_list, cluster_accuracies_global_list, cluster_accuracies_ref_list, 
                   score_moe_list, score_moe_list_glob, score_ref_moe_list, score_ref_nn_list, expert_range, 
                   save_path, dataset, missing, score_type, bandwidth, threshold_samples):
    """
    Compare the scores of MoE and reference models and create two separate plots.

    Parameters
    ----------
    ... (other parameters)

    Returns
    -------
    None
    """

    # Plot for Scores
    plt.figure(figsize=(10, 6))
    plt.plot(expert_range, score_moe_list, marker='o', label='MoE Local', color="blue")
    plt.plot(expert_range, score_moe_list_glob, marker='o', label='MoE Global', color="darkblue")
    plt.plot(expert_range, score_ref_moe_list, marker='o', label='Reference MoE', color="red")
    plt.plot(expert_range, [score_ref_nn_list] * len(expert_range), linestyle='--', marker='o', label='Reference NN', color="black")
    # Annotate score values above each x point
    # Annotate score values above each x point
    for i, txt in enumerate(score_moe_list):
        plt.annotate(f'{txt:.2f}', (expert_range[i], score_moe_list[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
    for i, txt in enumerate(score_moe_list_glob):
        plt.annotate(f'{txt:.2f}', (expert_range[i], score_moe_list_glob[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
    for i, txt in enumerate(score_ref_moe_list):
        plt.annotate(f'{txt:.2f}', (expert_range[i], score_ref_moe_list[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')

    
    plt.xlabel('Number of Experts')
    plt.ylabel('{}'.format(score_type))
    plt.title('Comparison of MoE and References {} for {}, Missing: {}, Threshold: {}'.format(score_type, dataset.upper(), missing, threshold_samples))
    plt.xticks(expert_range)
    plt.legend()
    plt.grid(True)

    # Create the full file path using os.path.join
    score_filename = os.path.join(save_path, "Scores of {} Missing {} Bandwidth {} Threshold {}.png".format(dataset.replace(":", "_").upper(), missing, bandwidth, threshold_samples))
    plt.savefig(score_filename)
    plt.show()
    plt.close()

    # Plot for Cluster Accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(expert_range, cluster_accuracies_local_list, marker='X', label='Local Cluster', color="blue", linestyle="dashed")
    plt.plot(expert_range, cluster_accuracies_global_list, marker='X', label='Global Cluster', color="darkblue", linestyle="dashed")
    plt.plot(expert_range, cluster_accuracies_ref_list, marker='X', label='Ref Cluster', color="red", linestyle="dashed")
    
    # Mark highest score with a larger yellow star
    max_score_idx_local = np.argmax(score_moe_list)
    max_score_idx_glob = np.argmax(score_moe_list_glob)
    max_score_idx_ref = np.argmax(score_ref_moe_list)
    star_size = 400  # Adjust the size of the star marker
    
    plt.scatter(expert_range[max_score_idx_local], cluster_accuracies_local_list[max_score_idx_local], marker='*', color='orange', s=star_size, label='Max Score MoE Local')
    plt.scatter(expert_range[max_score_idx_glob], cluster_accuracies_global_list[max_score_idx_glob], marker='*', color='orange', s=star_size, label='Max Score MoE Global')
    plt.scatter(expert_range[max_score_idx_ref], cluster_accuracies_ref_list[max_score_idx_ref], marker='*', color='orange', s=star_size, label='Max Score Ref MoE')
    
    # Annotate score values above each x point
    for i, txt in enumerate(score_moe_list):
        plt.annotate(f'{txt:.2f}', (expert_range[i], cluster_accuracies_local_list[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
    for i, txt in enumerate(score_moe_list_glob):
        plt.annotate(f'{txt:.2f}', (expert_range[i], cluster_accuracies_global_list[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
    for i, txt in enumerate(score_ref_moe_list):
        plt.annotate(f'{txt:.2f}', (expert_range[i], cluster_accuracies_ref_list[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')

    plt.xlabel('Number of Experts')
    plt.ylabel('Cluster Accuracies')
    plt.title('Comparison of Cluster Accuracies for {}, Missing: {}, Threshold: {}'.format(dataset.upper(), missing, threshold_samples))
    plt.xticks(expert_range)
    plt.legend()
    plt.grid(True)
    
    # Create the full file path using os.path.join
    cluster_acc_filename = os.path.join(save_path, "Cluster Accuracies of {} Missing {} Bandwidth {} Threshold {}.png".format(dataset.replace(":", "_").upper(), missing, bandwidth, threshold_samples))
    plt.savefig(cluster_acc_filename)
    plt.show()
    plt.close()
    
def delete_randomly_data(data, delete_percent, random_state=None):
    num_values_to_delete = int(data.size * delete_percent)
    np.random.seed(random_state)
    indices = np.random.choice(data.size, num_values_to_delete, replace=False)
    
    data_copy = data.flatten()
    if data_copy.dtype.kind == 'f':
        data_copy[indices] = np.nan
    else:
        data_copy = data_copy.astype(float)
        data_copy[indices] = np.nan
    
    return data_copy.reshape(data.shape)

def storeKeyValuePairs(*args):
    """
    Stores key-value pairs in the order they are received.

    Args:
        *args: Variable number of arguments as key-value pairs.

    Returns:
        dict: A dictionary containing the stored key-value pairs.
    """
    results_dict = {}
    for i in range(0, len(args), 2):
        if i + 1 < len(args):
            key = args[i]
            value = args[i + 1]
            results_dict[key] = value
    return results_dict


def saveResults(result_dict, folder, base_filename):
    """
    Saves measurement results from a dictionary to a CSV file.

    Args:
        result_dict (dict): A dictionary containing measurement names as keys and their corresponding results as values.
        folder (str): The folder path where the CSV file will be saved.
        base_filename (str): The base name of the CSV file (without the ".csv" extension).

    Returns:
        None
    """
    try:
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = os.path.join(folder, f'{base_filename}.csv')
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['Measurement', 'Result']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for measurement, result in result_dict.items():
                writer.writerow({'Measurement': measurement, 'Result': result})
        print(f'Results successfully saved to {file_path}.')
    except Exception as e:
        print(f'Error while saving results: {e}')
        
        
def find_best_expert(expert_counts, average_losses):
    best_expert_count = expert_counts[np.argmin(average_losses)]
    return best_expert_count


def plot_loss_vs_experts_fold(fold_num, file_path, local_moe_loss, global_moe_loss, ref_moe_mode_loss, ref_moe_ev_loss):
    experts = list(range(2, len(local_moe_loss) + 2))
    
    plt.plot(experts, local_moe_loss, marker='o', label='Local MoE')
    plt.plot(experts, global_moe_loss, marker='o', label='Global MoE')
    plt.plot(experts, ref_moe_mode_loss, marker='o', label='Ref. MoE Mode')
    plt.plot(experts, ref_moe_ev_loss, marker='o', label='Ref. MoE EV')
    
    plt.xlabel('Number of Experts')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss vs Number of Experts')
    plt.legend()
    plt.xticks(experts)  # Set the x-axis ticks to match the expert values

    
    # Ensure the directory exists before saving
    path = os.path.join(file_path, f"Val. Loss for Fold {fold_num}.png")
    plt.savefig(path)
    plt.close()



if __name__ == "__main__":

    
    pass