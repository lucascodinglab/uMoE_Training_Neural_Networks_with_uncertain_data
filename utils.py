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
        data = data.values
        X = data.drop(columns='microorganisms')
        y = np.array(data["microorganisms"])
        input_size = X.shape[1]
        output_size = 5
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
     
    if output_size > 1:
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder()
        y_labels = label_encoder.fit_transform(y)
        y = onehot_encoder.fit_transform(y_labels.reshape(-1, 1)).toarray()
        
    return X, y, input_size, output_size, score_type



def compare_scores(score_moe_list, score_ref_moe_list, score_ref_nn_list, expert_range, save_path, dataset, missing, score_type, bandwidth, threshold_samples):
    """
    Compare the scores of MoE and reference models and create a plot.

    Parameters
    ----------
    score_moe_list : list
        List of scores for MoE model for different numbers of experts.
    score_ref_moe_list : list
        List of scores for reference MoE model for different numbers of experts.
    score_ref_nn_list : list
        List of scores for reference NN model.
    expert_range : list
        List of integers representing the number of experts used.
    save_path : str
        Location where plot will be saved (including the folder path).
    dataset: str
        Name of the dataset
    Missing: float
        Percentage of Missing Values

    Returns
    -------
    None
        The function creates a plot to compare the scores.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(expert_range, score_moe_list, marker='o', label='MoE')
    plt.plot(expert_range, score_ref_moe_list, marker='o', label='Reference MoE')
    plt.plot(expert_range, [score_ref_nn_list[0]] * len(expert_range), linestyle='--', marker='o', label='Reference NN')

    plt.xlabel('Number of Experts')
    plt.ylabel('{}'.format(score_type))
    plt.title('Comparison of MoE and References {} for {}, Missing: {}, Threshold: {}'.format(score_type, dataset.upper(), missing, threshold_samples))
    plt.xticks(expert_range)
    plt.legend()
    plt.grid(True)

    # Remove the backslash from the beginning of save_path
    save_path = save_path.lstrip(os.path.sep)

    # Create the full file path using os.path.join
    filename = os.path.join(save_path, "{} of {} Missing {} Bandwidth {} Threshold {}.png".format(score_type, dataset.replace(":", "_").upper(), missing, bandwidth, threshold_samples))
    plt.savefig(filename)
    plt.show()
    plt.close()




if __name__ == "__main__":
    compare_scores([0.2,0.4], [0.2,0.5], [0.3,0.3], (1,2), r"D:\Github_Projects\Evaluation", "califronia", 0.3, "MSE",0.1,0.5)




    pass