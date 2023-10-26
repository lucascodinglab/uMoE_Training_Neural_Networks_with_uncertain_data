import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
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
    elif dataset == "banana":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv")
        X = data.drop(columns="Class", axis=1).values
        y = np.array(data["Class"].map({-1: 0, 1:1}))
        input_size = X.shape[1]
        output_size = 1
        score_type = "Accuracy"  

        
    if output_size > 1:
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder()
        y_labels = label_encoder.fit_transform(y)
        y = onehot_encoder.fit_transform(y_labels.reshape(-1, 1)).toarray()
        
    return X, y, input_size, output_size, score_type



def compare_scores( 
                   score_umoe_list, score_ref_moe_mode_list, score_ref_moe_ev_list, score_ref_nn_mode_list,
                   score_ref_nn_ev_list, expert_range, save_path, dataset, missing, score_type, bandwidth, threshold_samples):
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
    plt.plot(expert_range, score_umoe_list, marker='o', label='uMoE', color="blue")
    plt.plot(expert_range, score_ref_moe_mode_list, marker='o', label='Ref. MoE (Mode)', color="red")
    plt.plot(expert_range, score_ref_moe_ev_list, marker='o', label='Ref. MoE (EV)', color="#FF6666")
    plt.plot(expert_range, [score_ref_nn_mode_list] * len(expert_range), linestyle='--', marker='o', label='Ref. NN (Mode)', color="black")
    plt.plot(expert_range, [score_ref_nn_ev_list] * len(expert_range), linestyle='--', marker='o', label='Ref. NN (EV)', color="gray")

    for i, txt in enumerate(score_umoe_list):
        plt.annotate(f'{txt:.2f}', (expert_range[i], score_umoe_list[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
    
    # Annotate score values above each x point for Ref. MoE (Mode) scores
    for i, txt in enumerate(score_ref_moe_mode_list):
        plt.annotate(f'{txt:.2f}', (expert_range[i], score_ref_moe_mode_list[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
    
    # Annotate score values above each x point for Ref. MoE (EV) scores
    for i, txt in enumerate(score_ref_moe_ev_list):
        plt.annotate(f'{txt:.2f}', (expert_range[i], score_ref_moe_ev_list[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
    

    plt.xlabel('Number of Experts')
    plt.ylabel('{}'.format(score_type))
    plt.title('uMoE and References {} for {} (Uncertainty = {}, p = {})'.format(score_type, dataset.upper(), missing, threshold_samples))
    plt.xticks(expert_range)
    plt.legend()
    plt.grid(True)

    # Create the full file path using os.path.join
    score_filename = os.path.join(save_path, "Scores of {} Missing {} Bandwidth {} Threshold {}.png".format(dataset.replace(":", "_").upper(), missing, bandwidth, threshold_samples))
    plt.savefig(score_filename)
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

        
        
def find_best_expert(expert_counts, average_losses):
    best_expert_count = expert_counts[np.argmin(average_losses)]
    return best_expert_count




if __name__ == "__main__":
    


    pass