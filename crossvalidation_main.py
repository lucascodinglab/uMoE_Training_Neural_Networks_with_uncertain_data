import os
desired_directory = r"D:\Github_Projects\MOE_Training_under_Uncertainty"
os.chdir(desired_directory)

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold

import Prob_MoE as pm
import utils as ut
import uframe as uf

if __name__ == "__main__":
    
    
    dataset = "breast_cancer"
    result_path = r"D:\Github_Projects\Evaluation"
    missing = 0.2
    
    # Load data
    data, target, input_size, output_size = ut.preprocess_data(dataset=dataset)
    
    # Scale data
    size = 100
    data_sc = MinMaxScaler().fit_transform(data[:size])
    target = target[:size]
    
    # Specify the number of folds (k)
    n_folds = 5
    indices = np.arange(len(data_sc))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Uframe
    X = uf.uframe_from_array_mice_2(data_sc, kernel = "gaussian" , p =.1, mice_iterations = 2, bandwidth = 0.1)

    # select parameters
    n_experten_max = 4
    lr = 0.001
    reg_lambda = 0.0002
    batch_size_experts = 5
    batch_size_gate = 5
    n_epochs = 50
    threshold_samples = 0.5 # our method
    n_samples = 200 # our method
    local_mode = True # our method

    expert_range = range(2, n_experten_max + 1)
    score_moe_list = []
    score_ref_moe_list = []
    score_ref_nn_list = []
    
    for n in expert_range:
        # Print the indices for each fold
        for fold, (train_indices, test_indices) in enumerate(kf.split(indices)):
            val_size = len(train_indices) // 4  # 20% of train data for validation
            val_indices = train_indices[:val_size]
            train_indices = train_indices[val_size:]
            X_train = X[train_indices]
            target_train = target[train_indices]
            X_val = X[val_indices]
            target_val = target[val_indices]
            data_test = data[test_indices]
            target_test = target[test_indices]

            ########################### MoE #############################################################
            # MoE
            moe = pm.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [32, 32], hidden_gate = [32, 32])
            # val
            moe.fit(X_train, target_train, X_val, target_val, threshold_samples=threshold_samples, local_mode = local_mode, weighted_experts=True, 
                    verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
              
            # predictions / eval
            predictions = moe.predict(data_test)
            score_moe = moe.evaluation(predictions, target_test)   
            score_moe_list.append(score_moe)
    
            
            ############################ Referenz MoE #############################################################
            

            ref_train = uf.uframe()
            ref_train.append(X_train.mode())   
            
            
            # Ref MoE
            ref_moe = pm.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [32, 32], hidden_gate = [32, 32])
            # val
            ref_moe.fit(ref_train, target_train, X_val, target_val, threshold_samples=1, local_mode = False, weighted_experts=False, 
                    verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=1, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
              
            # predictions / eval
            predictions_ref = ref_moe.predict(data_test)
            score_ref_moe = ref_moe.evaluation(predictions_ref, target_test)
            score_ref_moe_list.append(score_ref_moe)
    
    
            
            ############################ Referenz NN #############################################################
            

            ref_train = uf.uframe()
            ref_train.append(X_train.mode())   
            
            
            # NN
            nn = pm.MoE(1, inputsize = input_size, outputsize = output_size, hidden_experts = [32, 32, 32],  hidden_gate=[1])
            # val
            nn.fit(ref_train, target_train, X_val, target_val, threshold_samples=1, local_mode = False, weighted_experts=False, 
                    verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=1, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
              
            # predictions / eval
            predictions_ref = nn.predict(data_test)
            score_ref_nn = nn.evaluation(predictions_ref, target_test)
            score_ref_nn_list.append(score_ref_nn)
        
    # plot results   
    ut.compare_scores(score_moe_list, score_ref_moe_list, [score_ref_nn]*len(expert_range), expert_range, result_path, dataset, missing)


 
        
        
        