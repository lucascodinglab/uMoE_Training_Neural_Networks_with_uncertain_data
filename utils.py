import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def preprocess_data(dataset = None):
    """
    

    Parameters
    ----------
    dataset : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    input_size : TYPE
        DESCRIPTION.
    output_size : TYPE
        DESCRIPTION.

    """
    data_path = r"D:\Github_Projects\Datasets"
    if dataset == "diabetes":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv") 
        X = data.drop(columns='Outcome')
        y = data["Outcome"]
        y = y.reset_index(drop=True)
        input_size = X.shape[1]
        output_size = 2
    elif dataset == "breast_cancer":
        data = pd.read_csv(data_path + "\\" + dataset + ".csv") 
        data.drop(columns=["id"],axis=1,inplace=True)
        X = data.drop(columns='diagnosis')
        y = data["diagnosis"]
        y = y.reset_index(drop=True)
        y = y.map({'M': 0, 'B': 1})
        input_size = X.shape[1]
        output_size = 2
    elif dataset == "california":
        data = fetch_california_housing()
        indices = np.arange(data.data.shape[0])
        np.random.shuffle(indices)
        X = data.data[indices]
        y = data.target[indices]
        input_size = X.shape[1]
        output_size = 1

    return X, y, input_size, output_size