import os
desired_directory = r"D:\Github_Projects\MOE_Training_under_Uncertainty"
os.chdir(desired_directory)

import numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# test
import Prob_MoE as pm
import utils as ut
import uframe as uf

if __name__ == "__main__":
    
    dataset = "wifi"
    # data_path = r"C:\Users\lul03615\Datasets"
    data_path = r"D:\Github_Projects\Datasets"
    result_path = r"D:\Github_Projects\Evaluation\wifi\Silverman"
    # load data
    data, target, input_size, output_size, score_type = ut.preprocess_data(data_path = data_path, dataset=dataset)
    
    
    size = len(data)
    data_sc = MinMaxScaler().fit_transform(data[:size])
    target = target[:size]
    
    
    # select setting
    missing = 0.5
    bandwidth = 0.01
    n_experts_max = 6
    # select parameters
    lr = 0.001
    reg_lambda = 0.0002
    batch_size_experts = 5
    batch_size_gate = 5
    n_epochs = 50
    threshold_samples = 0.5 # our method
    n_samples = 200 # our method
    local_mode = True # our method
    
    
    # split
    data_train, data_test, target_train, target_test = train_test_split(data_sc, target, test_size=0.2, random_state=42)
    # val
    data_train, data_val, target_train, target_val = train_test_split(data_train, target_train, test_size=0.25, random_state=42)   
    
    # uncertainty in data 
    X_train = uf.uframe_from_array_mice_2(data_train, kernel = "gaussian" , p =missing, mice_iterations = 2, bandwidth = bandwidth)
    X_val = uf.uframe_from_array_mice_2(data_val, kernel = "gaussian" , p =missing, mice_iterations = 2, bandwidth = bandwidth)
    # X.analysis(X_train, save= "filename", bins = 20)
    

    # save results
    expert_range = range(1, n_experts_max + 1)
    score_moe_list = []
    score_ref_moe_list = []
    score_ref_nn_list = []
    for n in expert_range:
        
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
        result_path_moe = result_path + "\MoE" + "_" + dataset + "_" + str(missing) + "_" + str(n) + "_" + str(bandwidth)
        moe.analyze(data_train, save_path = result_path_moe)    
        
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
        result_path_moe_ref = result_path + "\Ref_MoE" + "_" + dataset + "_" + str(missing) + "_" + str(n) + "_" + str(bandwidth)
        ref_moe.analyze(data_train, save_path = result_path_moe_ref)


        
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
    
    
    # plot results   
    ut.compare_scores(score_moe_list, score_ref_moe_list, score_ref_nn_list, 
                      expert_range, result_path, dataset, missing, score_type, bandwidth, threshold_samples)
    

 
    
    
    






    
        
    




