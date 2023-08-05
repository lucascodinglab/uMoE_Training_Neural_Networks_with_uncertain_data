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
    
    
    dataset = "california"
    data_path = r"D:\Github_Projects\Datasets"
    result_path = r"D:\Github_Projects\Evaluation"
    # select setting
    missing = 0.3
    bandwidth = 0.1
    n_folds = 2
    n_experts_max = 2
    # select parameters
    lr = 0.001
    reg_lambda = 0.0002
    batch_size_experts = 5
    batch_size_gate = 5
    n_epochs = 50
    threshold_samples = 0.5 # our method
    n_samples = 200 # our method
    local_mode = True # our method
    
    # Load data
    data, target, input_size, output_size, score_type = ut.preprocess_data(data_path = data_path, dataset=dataset)
    
    # Scale data
    size = 1000
    data_sc = MinMaxScaler().fit_transform(data[:size])
    target = target[:size]
    
    # Uframe
    X = uf.uframe_from_array_mice_2(data_sc, kernel = "gaussian" , p = missing, mice_iterations = 2, bandwidth = bandwidth)
    # X.analysis(X_train, save= dataset, bins = 20)


    ################################################## Crossvalidation ########################################################
    indices = np.arange(len(data_sc))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # save results
    expert_range = range(1, n_experts_max + 1)
    score_moe_list_total = []
    score_ref_moe_list_total = []
    # start crossvalidation
    for n in expert_range:
        score_moe_list = []
        score_ref_moe_list = []
        score_ref_nn_list = []
        # Print the indices for each fold
        for fold, (train_indices, test_indices) in enumerate(kf.split(indices)):
            val_size = len(train_indices) // 4  # 20% of train data for validation
            val_indices = train_indices[:val_size]
            train_indices = train_indices[val_size:]
            # split Uframe
            X_train = X[train_indices]
            target_train = target[train_indices]
            X_val = X[val_indices]
            target_val = target[val_indices]
            data_test = data[test_indices]
            target_test = target[test_indices]
            # global mode value for reference methods
            ref_train = uf.uframe()
            ref_train.append(X_train.mode()) 

            ########################### Prob. MoE #############################################################
            # MoE
            moe = pm.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [32, 32], hidden_gate = [32, 32])
            # val
            moe.fit(X_train, target_train, X_val, target_val, threshold_samples=threshold_samples, local_mode = local_mode, weighted_experts=True, 
                    verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
              
            # predictions / eval
            predictions = moe.predict(data_test)
            score_moe = moe.evaluation(predictions, target_test) 
            print(f"Prob MoE score: {score_moe}")
            score_moe_list.append(score_moe)
            # analyze
            result_path_moe = result_path + "\MoE" + "_" + dataset + "_" + str(missing) + "_" + str(n)
            moe.analyze(data_sc[train_indices], save_path = result_path_moe)    
            
            ############################ Referenz MoE #############################################################
            
            
            # Ref MoE
            ref_moe = pm.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [32, 32], hidden_gate = [32, 32])
            # val
            ref_moe.fit(ref_train, target_train, X_val, target_val, threshold_samples=1, local_mode = False, weighted_experts=False, 
                    verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=1, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
              
            # predictions / eval
            predictions_ref = ref_moe.predict(data_test)
            score_ref_moe = ref_moe.evaluation(predictions_ref, target_test)
            print(f"Ref MoE score: {score_ref_moe}")
            score_ref_moe_list.append(score_ref_moe)
            # analyze
            result_path_moe_ref = result_path + "\Ref_MoE" + "_" + dataset + "_" + str(missing) + "_" + str(n)
            ref_moe.analyze(data_sc[train_indices], save_path = result_path_moe)
    
    
            
            ############################ Referenz NN #############################################################
            if n == 1:
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
        
        # calculate average for every fold
        score_moe_list_total.append(np.mean(score_moe_list))
        score_ref_moe_list_total.append(np.mean(score_ref_moe_list))
        
    # plot results   
    ut.compare_scores(score_moe_list_total, score_ref_moe_list_total, [np.mean(score_ref_nn_list)] * n_experts_max, 
                      expert_range, result_path, dataset, missing, score_type, bandwidth, threshold_samples)


 
        
        
        