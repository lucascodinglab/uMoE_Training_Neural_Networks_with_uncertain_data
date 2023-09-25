import os
os.chdir(r"D:\Github_Projects\MOE_Training_under_Uncertainty")
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, KFold
# test
import uMoE as pm
import refMoE as rm
import utils as ut
import uframe as uf

if __name__ == "__main__":
    
    dataset = "breast_cancer"
    data_path = r"D:\Github_Projects\Datasets"
    result_path = r"D:\Evaluation\breast_cancer\u=0.4_p=0.8_b=0.1"
    # select setting
    missing = 0.4
    bandwidth = 0.1
    n_folds = 4
    n_experts_max = 6
    # select parameters
    lr = 0.001
    reg_lambda = 0.0002
    batch_size_experts = 16
    batch_size_gate = 32
    n_epochs = 100
    threshold_samples = 0.8 # our method
    n_samples = 100 # our method
    local_mode = True # our method
    
    # Load data
    data, target, input_size, output_size, score_type = ut.preprocess_data(data_path = data_path, dataset=dataset)

    # random_indices = np.random.choice(data.shape[0], 4000, replace=False)
    # data = data[random_indices]
    # target = target[random_indices]

    # Scale data
    data_sc = MinMaxScaler().fit_transform(data)
    target = target
    
    # Uframe
    X = uf.uframe_from_array_mice_2(data_sc, kernel = "gaussian" , p = missing, bandwidth = bandwidth)
    kwargs = {
          'stepsize':0.2,
          'niter':30,
    }  
    X_object = X.mode(**kwargs)
    # X.analysis(data_sc, save= data_path, bins = 20)


    ################################################## Crossvalidation ########################################################
    indices = np.arange(len(data_sc))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # save results
    expert_range = range(2, n_experts_max + 1)
    score_umoe_list_total = []
    score_ref_moe_mode_list_total = []
    score_ref_moe_ev_list_total = []
    score_ref_nn_mode_list_total = []
    score_ref_nn_ev_list_total = []
    # start crossvalidation
    for n in expert_range:
        score_umoe_list = []
        score_ref_moe_mode_list = []
        score_ref_moe_ev_list = []
        score_ref_nn_mode_list = []
        score_ref_nn_ev_list = []
        # Print the indices for each fold
        for fold, (train_indices, test_indices) in enumerate(kf.split(indices)):
            val_size = len(train_indices) // 3  # 20% of train data for validation
            val_indices = train_indices[:val_size]
            train_indices = train_indices[val_size:]
            # split Uframe
            X_train = X[train_indices]
            target_train = target[train_indices]
            X_val = X[val_indices]
            target_val = target[val_indices]
            data_test = data_sc[test_indices]
            target_test = target[test_indices]
            # global mode value for reference methods
            ref_train_mode = uf.uframe()
            ref_train_mode.append(X_train.mode()) 
            ref_train_ev = uf.uframe()
            ref_train_ev.append(X_train.ev()) 
            
    
            ########################### Prob. MoE #############################################################
            try:
                umoe = pm.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                umoe.fit(X_train, target_train, X_val, target_val, threshold_samples=threshold_samples, local_mode = local_mode, weighted_experts=True, 
                        verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                        n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
            except:
                umoe = pm.MoE(2, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                umoe.fit(X_train, target_train, X_val, target_val, threshold_samples=threshold_samples, local_mode = local_mode, weighted_experts=True, 
                        verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                        n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
            predictions = umoe.predict(data_test)
            score_moe = umoe.evaluation(predictions, target_test) 
            print(f"uMoE score: {score_moe}")
            score_umoe_list.append(score_moe)

            ############################ Referenz MoE MODE #############################################################
            try:
                ref_moe = rm.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                ref_moe.fit(ref_train_mode, target_train, X_val, target_val,
                            verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
            except:
                ref_moe = rm.MoE(2, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                ref_moe.fit(ref_train_mode, target_train, X_val, target_val,
                            verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
          
            # predictions / eval
            predictions_ref = ref_moe.predict(data_test)
            score_ref_moe = ref_moe.evaluation(predictions_ref, target_test)
            print(f"Ref Mode MoE score: {score_ref_moe}")
            score_ref_moe_mode_list.append(score_ref_moe)
   
            ############################ Referenz MoE EV #############################################################
            try:
                ref_moe = rm.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                ref_moe.fit(ref_train_ev, target_train, X_val, target_val,
                            verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
            except:
                ref_moe = rm.MoE(2, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                ref_moe.fit(ref_train_ev, target_train, X_val, target_val,
                            verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
            # predictions / eval
            predictions_ref = ref_moe.predict(data_test)
            score_ref_moe = ref_moe.evaluation(predictions_ref, target_test)
            print(f"Ref EV MoE score: {score_ref_moe}")
            score_ref_moe_ev_list.append(score_ref_moe)    
            
            ############################ Referenz NN Mode #############################################################
            if n == 2:
               # NN
                nn = rm.MoE(1, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16],  hidden_gate=[1])
                # val
                nn.fit(ref_train_mode, target_train, X_val, target_val,
                            verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
                  
                predictions_ref = nn.predict(data_test)
                score_ref_nn = nn.evaluation(predictions_ref, target_test)
                score_ref_nn_mode_list_total.append(score_ref_nn)
                print(f"Ref NN Mode score: {score_ref_nn}")
                
                
            ############################ Referenz NN Mode #############################################################
            if n == 2:
               # NN
                nn = rm.MoE(1, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16],  hidden_gate=[1])
                # val
                nn.fit(ref_train_ev, target_train, X_val, target_val,
                            verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
                  
                predictions_ref = nn.predict(data_test)
                score_ref_nn = nn.evaluation(predictions_ref, target_test)
                score_ref_nn_ev_list_total.append(score_ref_nn)
                print(f"Ref NN EV score: {score_ref_nn}")

           ########################################################################      
        # calculate average for every fold
        score_umoe_list_total.append(np.mean(score_umoe_list))
        score_ref_moe_mode_list_total.append(np.mean(score_ref_moe_mode_list))
        score_ref_moe_ev_list_total.append(np.mean(score_ref_moe_ev_list))

    score_ref_nn_ev_list_total = np.mean(score_ref_nn_ev_list_total)
    score_ref_nn_mode_list_total = np.mean(score_ref_nn_mode_list_total)
    ut.compare_scores(score_umoe_list_total, score_ref_moe_mode_list_total, score_ref_moe_ev_list_total, score_ref_nn_mode_list_total, score_ref_nn_ev_list_total, expert_range, result_path, dataset, missing, score_type, bandwidth, threshold_samples)

